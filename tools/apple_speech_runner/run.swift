import AVFoundation
import Foundation
import Speech

struct RunnerArguments {
    let modelID: String
    let audioPath: String
    let warmupAudioPath: String?
    let sampleSeconds: Double
    let localeIdentifier: String
    let warmups: Int
    let runs: Int
    let auto: Bool
    let autoMinRuns: Int
    let autoMaxRuns: Int
    let autoTargetCV: Double
}

enum RunnerError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case unsupportedModel(String)
    case unavailable(String)
    case unsupportedLocale(String)

    var description: String {
        switch self {
        case .invalidArguments(let message):
            return message
        case .unsupportedModel(let modelID):
            return "unsupported model id: \(modelID)"
        case .unavailable(let message):
            return message
        case .unsupportedLocale(let locale):
            return "unsupported locale: \(locale)"
        }
    }
}

func parseArguments() throws -> RunnerArguments {
    var values: [String: String] = [:]
    var flags: Set<String> = []
    var index = 1
    while index < CommandLine.arguments.count {
        let arg = CommandLine.arguments[index]
        if arg.hasPrefix("--") {
            if index + 1 < CommandLine.arguments.count,
               !CommandLine.arguments[index + 1].hasPrefix("--") {
                values[arg] = CommandLine.arguments[index + 1]
                index += 2
            } else {
                flags.insert(arg)
                index += 1
            }
        } else {
            index += 1
        }
    }

    guard let modelID = values["--model-id"] else {
        throw RunnerError.invalidArguments("missing --model-id")
    }
    guard let audioPath = values["--audio-path"] else {
        throw RunnerError.invalidArguments("missing --audio-path")
    }
    guard let sampleSecondsRaw = values["--sample-seconds"],
          let sampleSeconds = Double(sampleSecondsRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --sample-seconds")
    }
    guard let localeIdentifier = values["--locale"], !localeIdentifier.isEmpty else {
        throw RunnerError.invalidArguments("missing --locale")
    }
    guard let warmupsRaw = values["--warmups"], let warmups = Int(warmupsRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --warmups")
    }
    guard let runsRaw = values["--runs"], let runs = Int(runsRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --runs")
    }
    guard let autoMinRaw = values["--auto-min-runs"], let autoMinRuns = Int(autoMinRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --auto-min-runs")
    }
    guard let autoMaxRaw = values["--auto-max-runs"], let autoMaxRuns = Int(autoMaxRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --auto-max-runs")
    }
    guard let autoTargetRaw = values["--auto-target-cv"], let autoTargetCV = Double(autoTargetRaw) else {
        throw RunnerError.invalidArguments("missing or invalid --auto-target-cv")
    }

    return RunnerArguments(
        modelID: modelID,
        audioPath: audioPath,
        warmupAudioPath: values["--warmup-audio-path"],
        sampleSeconds: sampleSeconds,
        localeIdentifier: localeIdentifier,
        warmups: warmups,
        runs: runs,
        auto: flags.contains("--auto"),
        autoMinRuns: autoMinRuns,
        autoMaxRuns: autoMaxRuns,
        autoTargetCV: autoTargetCV
    )
}

func cleanTranscript(_ text: AttributedString) -> String? {
    let string = String(text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
    return string.isEmpty ? nil : string
}

func resolveSupportedLocale(modelID: String, localeIdentifier: String) async throws -> Locale {
    guard modelID == "apple/speech-transcriber" else {
        throw RunnerError.unsupportedModel(modelID)
    }
    guard SpeechTranscriber.isAvailable else {
        throw RunnerError.unavailable("SpeechTranscriber is not available on this machine")
    }
    let requested = Locale(identifier: localeIdentifier)
    guard let supported = await SpeechTranscriber.supportedLocale(equivalentTo: requested) else {
        throw RunnerError.unsupportedLocale(localeIdentifier)
    }
    return supported
}

func makeTranscriber(locale: Locale) -> SpeechTranscriber {
    SpeechTranscriber(locale: locale, preset: .transcription)
}

func ensureAssets(locale: Locale) async throws {
    let modules: [any SpeechModule] = [makeTranscriber(locale: locale)]
    let status = await AssetInventory.status(forModules: modules)
    switch status {
    case .unsupported:
        throw RunnerError.unavailable("speech assets unsupported for selected locale")
    case .installed:
        return
    case .supported, .downloading:
        if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
            try await request.downloadAndInstall()
        }
    @unknown default:
        return
    }
}

func transcribeFile(path: String, locale: Locale) async throws -> String? {
    let transcriber = makeTranscriber(locale: locale)
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)
    let transcriptTask = Task { () throws -> String? in
        var parts: [String] = []
        for try await result in transcriber.results {
            if result.isFinal, let text = cleanTranscript(result.text) {
                parts.append(text)
            }
        }
        let joined = parts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? nil : joined
    }
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    _ = try await analyzer.analyzeSequence(from: file)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    return try await transcriptTask.value
}

@main
struct Main {
    static func main() async throws {
        let args = try parseArguments()
        let locale = try await resolveSupportedLocale(
            modelID: args.modelID,
            localeIdentifier: args.localeIdentifier
        )
        try await ensureAssets(locale: locale)

        if let warmupAudioPath = args.warmupAudioPath, !warmupAudioPath.isEmpty {
            for _ in 0..<args.warmups {
                _ = try await transcribeFile(path: warmupAudioPath, locale: locale)
            }
        } else {
            for _ in 0..<args.warmups {
                _ = try await transcribeFile(path: args.audioPath, locale: locale)
            }
        }

        var elapsedValues: [Double] = []
        var transcripts: [String?] = []

        func shouldStopAuto() -> Bool {
            let minRuns = max(1, args.autoMinRuns)
            if elapsedValues.count < minRuns {
                return false
            }
            let maxRuns = max(minRuns, args.autoMaxRuns)
            if elapsedValues.count >= maxRuns {
                return true
            }
            let mean = elapsedValues.reduce(0, +) / Double(elapsedValues.count)
            if mean <= 0 {
                return false
            }
            let variance = elapsedValues.reduce(0) { partial, value in
                let delta = value - mean
                return partial + (delta * delta)
            } / Double(max(1, elapsedValues.count - 1))
            let stdev = elapsedValues.count >= 2 ? sqrt(variance) : 0
            let cv = stdev / mean
            return cv <= max(0, args.autoTargetCV)
        }

        let wallStart = Date()
        while true {
            let start = Date()
            let transcript = try await transcribeFile(path: args.audioPath, locale: locale)
            let elapsed = Date().timeIntervalSince(start)
            elapsedValues.append(elapsed)
            transcripts.append(transcript)
            if !args.auto {
                if elapsedValues.count >= args.runs {
                    break
                }
            } else if shouldStopAuto() {
                break
            }
        }

        let rtfxValues = elapsedValues.map { $0 > 0 ? args.sampleSeconds / $0 : 0 }
        let rtfxMean = rtfxValues.reduce(0, +) / Double(max(1, rtfxValues.count))
        let rtfxStdev: Double = {
            guard rtfxValues.count >= 2 else { return 0 }
            let mean = rtfxMean
            let variance = rtfxValues.reduce(0) { partial, value in
                let delta = value - mean
                return partial + (delta * delta)
            } / Double(rtfxValues.count - 1)
            return sqrt(variance)
        }()

        let lastTranscript: Any = {
            if let last = transcripts.last, let transcript = last {
                return transcript
            }
            return NSNull()
        }()
        let transcriptValues: [Any] = transcripts.map { transcript in
            if let transcript {
                return transcript
            }
            return NSNull()
        }
        let payload: [String: Any] = [
            "rtfx_mean": rtfxMean,
            "rtfx_stdev": rtfxStdev,
            "wall_seconds": Date().timeIntervalSince(wallStart),
            "device": "system",
            "transcript": lastTranscript,
            "elapsed_values": elapsedValues,
            "transcripts": transcriptValues
        ]
        let json = try JSONSerialization.data(withJSONObject: payload, options: [])
        if let output = String(data: json, encoding: .utf8) {
            print(output)
        }
    }
}

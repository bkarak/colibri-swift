// Hand-written chat renderers, as upstream does (c/openai_server.py render_chat_*); the
// GGUF's Jinja template is never executed. docs/port/models.md section 7. ChatML follows
// the Qwen3 GGUF's own Jinja for plain chats (no tools); OLMoE follows
// render_chat_olmoe in c/openai_server.py.

import Foundation

public struct ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable { case system, user, assistant }
    public var role: Role
    public var content: String
    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Renders a conversation into the prompt text the model was trained on. A family that
/// needs a new template defines its own conforming type in its own file.
public protocol ChatTemplate: Sendable {
    /// The prompt for `messages` followed by the generation prompt for the next
    /// assistant turn.
    func render(_ messages: [ChatMessage]) -> String
}

/// ChatML as Qwen3 uses it, with the thinking switch, byte for byte the GGUF's Jinja
/// (`enable_thinking`) for chats without tools:
///
/// - a leading system message is `<|im_start|>system\n...<|im_end|>\n`, users likewise;
/// - an assistant turn after the last user query keeps its reasoning as
///   `<think>\n...\n</think>\n\n` (the last one always, an earlier one only if it has
///   reasoning); turns before it lose their `<think>` block;
/// - the generation prompt is `<|im_start|>assistant\n`, followed by the empty block
///   `<think>\n\n</think>\n\n` when thinking is off. Thinking on adds nothing: the model
///   opens `<think>` itself.
///
/// Soft switches: Qwen3 is trained to obey `/think` and `/no_think` written in a user or
/// system message (the text stays in the prompt). The most recent switch in the
/// conversation overrides `thinking` for the generation prompt, so `/no_think` gets the
/// empty block and `/think` lets the model reason even when the default is off.
///
/// `thinkingSupported == false` is the Qwen3-Coder layout: plain ChatML, no think lines,
/// assistant history verbatim.
public struct ChatMLTemplate: ChatTemplate {
    public var thinkingSupported: Bool
    public var thinking: Bool
    public init(thinkingSupported: Bool = true, thinking: Bool = false) {
        self.thinkingSupported = thinkingSupported
        self.thinking = thinking
    }

    /// The most recent `/think` or `/no_think` in a user or system message, if any
    /// (true for `/think`).
    public static func softSwitch(_ messages: [ChatMessage]) -> Bool? {
        for m in messages.reversed() where m.role != .assistant {
            let think = m.content.range(of: "/think", options: .backwards)?.lowerBound
            let noThink = m.content.range(of: "/no_think", options: .backwards)?.lowerBound
            switch (think, noThink) {
            case (nil, nil): continue
            case (.some, nil): return true
            case (nil, .some): return false
            case (.some(let a), .some(let b)): return a > b
            }
        }
        return nil
    }

    /// Whether the generation prompt leaves the model free to think.
    public func thinkingEnabled(for messages: [ChatMessage]) -> Bool {
        thinkingSupported && (ChatMLTemplate.softSwitch(messages) ?? thinking)
    }

    public func render(_ messages: [ChatMessage]) -> String {
        var out = ""
        // last_query_index: the last user message (no tool results in this renderer)
        let lastQuery = messages.lastIndex { $0.role == .user } ?? (messages.count - 1)
        for (index, m) in messages.enumerated() {
            switch m.role {
            case .system, .user:
                out += "<|im_start|>\(m.role.rawValue)\n\(m.content)<|im_end|>\n"
            case .assistant:
                guard thinkingSupported else {
                    out += "<|im_start|>assistant\n\(m.content)<|im_end|>\n"
                    continue
                }
                var content = m.content
                var reasoning = ""
                if let close = content.range(of: "</think>", options: .backwards) {
                    let before = content[..<content.range(of: "</think>")!.lowerBound]
                    var r = Substring(before).rstrip("\n")
                    if let open = r.range(of: "<think>", options: .backwards) { r = r[open.upperBound...] }
                    reasoning = String(r.lstrip("\n"))
                    content = String(content[close.upperBound...].lstrip("\n"))
                }
                if index > lastQuery && (index == messages.count - 1 || !reasoning.isEmpty) {
                    let r = Substring(reasoning).lstrip("\n").rstrip("\n")
                    out += "<|im_start|>assistant\n<think>\n\(r)\n</think>\n\n\(content.lstrip("\n"))<|im_end|>\n"
                } else {
                    out += "<|im_start|>assistant\n\(content)<|im_end|>\n"
                }
            }
        }
        out += "<|im_start|>assistant\n"
        if thinkingSupported && !thinkingEnabled(for: messages) {
            out += "<think>\n\n</think>\n\n"
        }
        return out
    }
}

/// OLMoE-Instruct (Tulu style, `|||IP_ADDRESS|||` as BOS/EOS text), as
/// c/openai_server.py render_chat_olmoe writes it:
/// `{bos}<|system|>\n{s}\n<|user|>\n{u}\n<|assistant|>\n{a}{eos}\n...<|assistant|>\n`.
public struct OLMoETemplate: ChatTemplate {
    /// The BOS/EOS text the template writes (OLMoE's tokenizer uses one for both).
    public var boundary: String
    public init(boundary: String = "|||IP_ADDRESS|||") {
        self.boundary = boundary
    }

    public func render(_ messages: [ChatMessage]) -> String {
        var out = boundary
        let last = messages.count - 1
        for (index, m) in messages.enumerated() {
            switch m.role {
            case .system: out += "<|system|>\n\(m.content)\n"
            case .user: out += "<|user|>\n\(m.content)\n"
            case .assistant:
                out += "<|assistant|>\n\(m.content)\(boundary)"
                if index != last { out += "\n" }
            }
        }
        out += "<|assistant|>\n"
        return out
    }
}

/// No template: the user text is the prompt (base models, oracle comparisons, `--raw`).
public struct RawTemplate: ChatTemplate {
    public init() {}
    public func render(_ messages: [ChatMessage]) -> String {
        messages.map(\.content).joined()
    }
}

extension Substring {
    /// Python's `lstrip(ch)` / `rstrip(ch)` for one character.
    fileprivate func lstrip(_ ch: Character) -> Substring {
        var s = self
        while s.first == ch { s = s.dropFirst() }
        return s
    }
    fileprivate func rstrip(_ ch: Character) -> Substring {
        var s = self
        while s.last == ch { s = s.dropLast() }
        return s
    }
}

extension String {
    fileprivate func lstrip(_ ch: Character) -> Substring { Substring(self).lstrip(ch) }
}

import Testing
@testable import ColibriCore

// Expected strings are what the Qwen3-0.6B GGUF's Jinja template (tokenizer.chat_template)
// renders for the same messages, worked through by hand; the OLMoE ones follow
// c/openai_server.py render_chat_olmoe.

@Suite struct ChatTemplateTests {
    let user = ChatMessage(role: .user, content: "Hi")

    @Test func qwen3ThinkingOff() {
        let t = ChatMLTemplate(thinking: false)
        #expect(t.render([user]) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func qwen3ThinkingOn() {
        let t = ChatMLTemplate(thinking: true)
        #expect(t.render([user]) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
    }

    @Test func qwen3SystemMessage() {
        let t = ChatMLTemplate(thinking: true)
        let m = [ChatMessage(role: .system, content: "Be brief."), user]
        #expect(t.render(m) == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
    }

    @Test func qwen3HistoryDropsOldReasoning() {
        let t = ChatMLTemplate(thinking: false)
        let m = [
            user,
            ChatMessage(role: .assistant, content: "<think>\nGreet back.\n</think>\n\nHello!"),
            ChatMessage(role: .user, content: "Bye"),
        ]
        #expect(t.render(m) == """
        <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n\
        <|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        """)
    }

    @Test func qwen3TrailingAssistantKeepsThinkBlock() {
        // an assistant turn after the last user query keeps its <think> block
        let t = ChatMLTemplate(thinking: true)
        let m = [user, ChatMessage(role: .assistant, content: "<think>\nplan\n</think>\n\nHello!")]
        #expect(t.render(m) == """
        <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nplan\n</think>\n\nHello!<|im_end|>\n\
        <|im_start|>assistant\n
        """)
        let plain = [user, ChatMessage(role: .assistant, content: "Hello!")]
        #expect(t.render(plain) == """
        <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n\
        <|im_start|>assistant\n
        """)
    }

    @Test func softSwitches() {
        let off = ChatMLTemplate(thinking: false)
        let on = ChatMLTemplate(thinking: true)
        let think = [ChatMessage(role: .user, content: "Prove it /think")]
        let noThink = [ChatMessage(role: .user, content: "Quick answer /no_think")]
        // the switch text stays in the prompt; it decides the generation prompt
        #expect(off.render(think) == "<|im_start|>user\nProve it /think<|im_end|>\n<|im_start|>assistant\n")
        #expect(on.render(noThink) == "<|im_start|>user\nQuick answer /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        // the most recent switch wins, across turns and within a message
        let both = [
            ChatMessage(role: .user, content: "a /no_think"),
            ChatMessage(role: .assistant, content: "ok"),
            ChatMessage(role: .user, content: "b /think"),
        ]
        #expect(ChatMLTemplate.softSwitch(both) == true)
        #expect(ChatMLTemplate.softSwitch([ChatMessage(role: .user, content: "/think then /no_think")]) == false)
        #expect(ChatMLTemplate.softSwitch([user]) == nil)
        #expect(on.thinkingEnabled(for: [user]))
        #expect(!off.thinkingEnabled(for: [user]))
    }

    @Test func qwen3CoderHasNoThinkLines() {
        let t = ChatMLTemplate(thinkingSupported: false, thinking: true)
        let m = [user, ChatMessage(role: .assistant, content: "Hello!"), ChatMessage(role: .user, content: "/no_think x")]
        #expect(t.render(m) == """
        <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n\
        <|im_start|>user\n/no_think x<|im_end|>\n<|im_start|>assistant\n
        """)
    }

    @Test func olmoe() {
        let t = OLMoETemplate()
        #expect(t.render([user]) == "|||IP_ADDRESS|||<|user|>\nHi\n<|assistant|>\n")
        let m = [
            ChatMessage(role: .system, content: "S"),
            user,
            ChatMessage(role: .assistant, content: "Hello!"),
            ChatMessage(role: .user, content: "Bye"),
        ]
        #expect(t.render(m) == """
        |||IP_ADDRESS|||<|system|>\nS\n<|user|>\nHi\n<|assistant|>\nHello!|||IP_ADDRESS|||\n\
        <|user|>\nBye\n<|assistant|>\n
        """)
    }

    @Test func raw() {
        #expect(RawTemplate().render([user]) == "Hi")
    }
}

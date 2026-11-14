import Foundation

/// The one error type the engine throws. The cases are deliberately coarse so that no
/// workstream has to add a case: put the detail in the message.
public enum ColibriError: Error, CustomStringConvertible, Equatable {
    /// A stubbed code path. Thrown (not trapped) where a caller can reasonably recover,
    /// for example the CLI reporting that a subcommand is not built yet.
    case notImplemented(String)
    /// The GGUF file is malformed or violates a sanity limit (untrusted input).
    case invalidGGUF(String)
    /// The file is well formed but uses something the port does not support
    /// (an unknown architecture, a ggml type without a decoder, a missing tensor).
    case unsupported(String)
    /// An operating-system call failed. `errno` is captured at the failure site.
    case io(path: String, operation: String, errno: Int32)
    /// A caller passed an argument that violates a documented precondition.
    case invalidArgument(String)
    /// A backend could not be created or failed while executing (Metal device missing,
    /// shader compile error, command buffer error).
    case backend(String)

    public var description: String {
        switch self {
        case .notImplemented(let what): return "not implemented: \(what)"
        case .invalidGGUF(let why): return "invalid GGUF: \(why)"
        case .unsupported(let why): return "unsupported: \(why)"
        case .io(let path, let op, let e):
            return "\(op) failed for \(path): \(String(cString: strerror(e))) (errno \(e))"
        case .invalidArgument(let why): return "invalid argument: \(why)"
        case .backend(let why): return "backend error: \(why)"
        }
    }
}

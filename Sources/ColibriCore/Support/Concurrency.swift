import Foundation

/// Carries a non-Sendable value (typically a raw pointer) into a
/// `DispatchQueue.concurrentPerform` closure or onto an I/O thread. The caller promises
/// that the accesses are race-free (disjoint output ranges, or read-only sharing).
public struct UnsafeSendable<T>: @unchecked Sendable {
    public var value: T
    @inlinable public init(_ value: T) { self.value = value }
}

/// Rounds `value` up to a multiple of `alignment` (a power of two).
@inlinable public func alignUp(_ value: Int, _ alignment: Int) -> Int {
    precondition(alignment > 0 && alignment & (alignment - 1) == 0, "alignment must be a power of two")
    return (value + alignment - 1) & ~(alignment - 1)
}

/// The Apple Silicon VM page size, and the alignment Metal requires for
/// `makeBuffer(bytesNoCopy:)` (both pointer and length).
public let hostPageSize = 16_384

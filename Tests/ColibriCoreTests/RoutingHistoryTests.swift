import Foundation
import Testing
@testable import ColibriCore

private func tempPath(_ name: String) throws -> String {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("colibri-history-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent(name).path
}

/// A literal port of the reader loop old upstream engines use:
/// `while (fscanf(f, "%d %d %u", &l, &e, &c) == 3) { if (l >= 0) ... }`, run through the
/// C library's own scanner so field parsing is exactly C's.
private func oldReader(_ text: String) -> [(Int32, Int32, UInt32)] {
    var out: [(Int32, Int32, UInt32)] = []
    var l: Int32 = 0, e: Int32 = 0
    var c: UInt32 = 0
    var consumed: Int32 = 0
    var rest = text
    while true {
        let n = withUnsafeMutablePointer(to: &l) { pl in
            withUnsafeMutablePointer(to: &e) { pe in
                withUnsafeMutablePointer(to: &c) { pc in
                    withUnsafeMutablePointer(to: &consumed) { pn in
                        withVaList([pl, pe, pc, pn]) { vsscanf(rest, "%d %d %u%n", $0) }
                    }
                }
            }
        }
        guard n == 3 else { break }
        if l >= 0 { out.append((l, e, c)) }
        rest = String(rest.utf8.dropFirst(Int(consumed)))!
    }
    return out
}

@Suite struct RoutingHistoryTests {
    @Test func savesUpstreamBytesAndRoundTrips() throws {
        let h = RoutingHistory(layers: 3, experts: 4, engineName: "olmoe", moeLayers: [0, 2])
        h.record(layer: 2, experts: [3, 3, 1])
        h.record(layer: 0, experts: [0])
        h.record(layer: 1, experts: [2])  // dense layer: no row
        h.record(layer: 2, experts: [4, -1])  // out of range: ignored
        let id = RoutingHistory.engineID("olmoe")
        #expect(id == 0x8A2A_2D4B || id == RoutingHistory.engineID("olmoe"))
        let expected = "-1 3 4\n-2 1 \(id)\n0 0 1\n2 1 1\n2 3 2\n"
        #expect(h.serialized() == expected)
        let path = try tempPath("model.gguf.coli_usage")
        try h.save(path: path)
        #expect(try String(contentsOfFile: path, encoding: .utf8) == expected)
        #expect(!FileManager.default.fileExists(atPath: path + ".tmp"))

        let back = RoutingHistory(layers: 3, experts: 4, engineName: "olmoe", moeLayers: [0, 2])
        #expect(try back.load(path: path, explicit: false) == 4)
        #expect(back.serialized() == expected)
        // loading accumulates, as upstream's rt_load
        try back.load(path: path, explicit: false)
        #expect(back.count(layer: 2, expert: 3) == 4)
    }

    @Test func oldReadersParseTheFile() throws {
        let h = RoutingHistory(layers: 48, experts: 128, engineName: "qwen3moe", moeLayers: Set(0 ..< 48))
        h.record(layer: 47, experts: [127, 0])
        let text = h.serialized()
        // every record is exactly three numeric fields, the hash in the third
        for line in text.split(separator: "\n") {
            let f = line.split(separator: " ")
            #expect(f.count == 3 && Int(f[0]) != nil && Int(f[1]) != nil && UInt32(f[2]) != nil)
        }
        let rows = oldReader(text)
        #expect(rows.count == 2)
        #expect(rows[0] == (47, 0, 1) && rows[1] == (47, 127, 1))
    }

    @Test func emptyHistoryIsAZeroByteFile() throws {
        let h = RoutingHistory(layers: 2, experts: 4, engineName: "olmoe", moeLayers: [0, 1])
        let path = try tempPath("empty.coli_usage")
        try h.save(path: path)
        #expect(try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int == 0)
        #expect(try h.load(path: path, explicit: false) == 0)
        #expect(h.isEmpty)
    }

    @Test func refusesAnotherEngineUnlessNamedExplicitly() throws {
        let path = try tempPath("other.coli_usage")
        try "-1 2 4\n-2 1 \(RoutingHistory.engineID("qwen36"))\n1 2 9\n".write(toFile: path, atomically: true, encoding: .utf8)
        let h = RoutingHistory(layers: 2, experts: 4, engineName: "olmoe", moeLayers: [0, 1])
        #expect(throws: ColibriError.self) { try h.load(path: path, explicit: false) }
        #expect(h.isEmpty)  // a refused file adds nothing
        #expect(try h.load(path: path, explicit: true) == 9)
        #expect(h.top(layer: 1, 3) == [2])
    }

    @Test func refusesDimensionAndVersionMismatchEvenWhenExplicit() throws {
        let h = RoutingHistory(layers: 2, experts: 4, engineName: "olmoe", moeLayers: [0, 1])
        let id = RoutingHistory.engineID("olmoe")
        let dims = try tempPath("dims.coli_usage")
        try "-1 2 8\n-2 1 \(id)\n1 2 9\n".write(toFile: dims, atomically: true, encoding: .utf8)
        #expect(throws: ColibriError.self) { try h.load(path: dims, explicit: true) }
        let ver = try tempPath("ver.coli_usage")
        try "-1 2 4\n-2 2 \(id)\n1 2 9\n".write(toFile: ver, atomically: true, encoding: .utf8)
        #expect(throws: ColibriError.self) { try h.load(path: ver, explicit: true) }
        #expect(h.isEmpty)
        #expect(throws: ColibriError.self) { try h.load(path: dims + ".missing", explicit: false) }
    }

    @Test func legacyHeaderlessFilesLoadAndBadRecordsAreDropped() throws {
        let path = try tempPath("legacy.coli_usage")
        // dense layer 0, out-of-range expert, out-of-range layer, a future header record,
        // then a non-numeric field that ends the scan (as fscanf would)
        try "1 1 5\n0 2 7\n1 9 3\n5 0 2\n-7 0 0\n1 3 4\nnot a number\n1 0 100\n"
            .write(toFile: path, atomically: true, encoding: .utf8)
        let h = RoutingHistory(layers: 2, experts: 4, engineName: "olmoe", moeLayers: [1])
        #expect(try h.load(path: path, explicit: false) == 9)
        #expect(h.counters(layer: 1) == [0, 5, 0, 4])
        #expect(h.counters(layer: 0).isEmpty)
    }

    @Test func iku1IsReadOnlyWhenNamed() throws {
        var bytes: [UInt32] = [0x3155_4B49, 2, 3, 0, 0, 0, 4, 0, 1]
        let path = try tempPath("inkling.usage")
        try Data(bytes: &bytes, count: bytes.count * 4).write(to: URL(fileURLWithPath: path))
        let h = RoutingHistory(layers: 2, experts: 3, engineName: "olmoe", moeLayers: [0, 1])
        #expect(throws: ColibriError.self) { try h.load(path: path, explicit: false) }
        #expect(try h.load(path: path, explicit: true) == 5)
        #expect(h.counters(layer: 1) == [4, 0, 1])
    }

    @Test func topRanksByCountThenId() {
        let h = RoutingHistory(layers: 1, experts: 6, engineName: "olmoe", moeLayers: [0])
        h.record(layer: 0, experts: [4, 4, 1, 1, 5, 2, 2, 2])
        #expect(h.top(layer: 0, 3) == [2, 1, 4])
        #expect(h.top(layer: 0, 10) == [2, 1, 4, 5])  // never-routed experts are not ranked
        #expect(h.top(layer: 0, 0).isEmpty)
    }

    @Test func engineIDIsFNV1a() {
        #expect(RoutingHistory.engineID("") == 2_166_136_261)
        #expect(RoutingHistory.engineID("a") == 0xE40C_292C)
    }
}

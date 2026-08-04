import Foundation

let url = URL(fileURLWithPath: "../zurich_world.json")
var t0 = Date()
let net = try RoadNetwork(contentsOf: url)
print(String(format: "Loaded %@: %d roads in %.2f s", net.name, net.roads.count, -t0.timeIntervalSinceNow))

// 1. Spawn on a real street.
guard let spawn = net.spawn(on: "Bahnhofstrasse") else { fatalError("no Bahnhofstrasse") }
print(String(format: "Spawn on Bahnhofstrasse at (%.1f, %.1f, %.1f) yaw %.2f rad",
             spawn.position.x, spawn.position.y, spawn.position.z, spawn.yaw))

// 2. The ground under the spawn must be asphalt at the road's height.
let s = net.sampleGround(x: spawn.position.x, z: spawn.position.z)
print(String(format: "  ground: %@ at y=%.2f (road y=%.2f, delta %.3f m)",
             "\(s.surface)", s.height, spawn.position.y, abs(s.height - spawn.position.y)))

// 3. Step PERPENDICULAR to the road and watch the surface change.
let perp = (x: cos(spawn.yaw), z: sin(spawn.yaw))
print("Crossing the road edge perpendicular to the centreline:")
for off in stride(from: 0.0, through: 9.0, by: 1.0) {
    let g = net.sampleGround(x: spawn.position.x + perp.x * off,
                             z: spawn.position.z + perp.z * off)
    print(String(format: "  +%4.1f m  %-8@ y=%.2f", off, "\(g.surface)", g.height))
}

// 4. Drive a straight line across the whole world and count what we find.
print("Sampling a 3 km transect across the city:")
var counts: [SurfaceKind: Int] = [:]
var minY = Double.infinity, maxY = -Double.infinity
for i in 0..<3000 {
    let x = -1500.0 + Double(i)
    let g = net.sampleGround(x: x, z: 0)
    counts[g.surface, default: 0] += 1
    minY = min(minY, g.height); maxY = max(maxY, g.height)
}
print("  surfaces: \(counts.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " "))")
print(String(format: "  height range %.1f..%.1f m", minY, maxY))

// 5. Throughput — this runs 4x per wheel per physics tick.
t0 = Date()
var sink = 0.0
let n = 400_000
for i in 0..<n {
    let a = Double(i % 3000) - 1500, b = Double((i / 3000) % 3000) - 1500
    sink += net.sampleGround(x: a, z: b).height
}
let dt = -t0.timeIntervalSinceNow
print(String(format: "Throughput: %d samples in %.2f s = %.2f M/s (sink %.0f)",
             n, dt, Double(n) / dt / 1e6, sink))

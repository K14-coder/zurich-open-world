import Foundation
import simd

/// Turns the exported Zurich world into triangles.
///
/// Everything is baked into one vertex buffer at load: the city is small enough
/// (a few hundred thousand triangles) that streaming and LOD would be effort
/// spent solving a problem this world does not have.
/// - Note: every field is a 16-byte-aligned slot, giving a 64-byte stride that
///   the shader's `float3, float3, float4, float4` matches exactly. Adding a
///   bare `Float` anywhere would silently change the stride and corrupt every
///   vertex — there is a `precondition` on this in main.swift.
struct Vertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var colour: SIMD4<Float>
    /// Per-building: base elevation, storey height, random seed, total height.
    /// Zero for terrain and roads. Without the base, storey lines are keyed to
    /// absolute world height and float free of the building they belong to.
    var params: SIMD4<Float>

    enum Material: Float {
        case plain = 0      // terrain, roads
        case wall = 1       // building façades — gets procedural windows
        case roof = 2
        case ortho = 3      // terrain — sampled from the aerial photograph
    }

    init(position: SIMD3<Float>, normal: SIMD3<Float>,
         colour: SIMD3<Float>, material: Material = .plain,
         params: SIMD4<Float> = .zero) {
        self.position = position
        self.normal = normal
        self.colour = SIMD4(colour, material.rawValue)
        self.params = params
    }
}

final class WorldMesh {
    private(set) var vertices: [Vertex] = []
    private(set) var indices: [UInt32] = []

    /// Where each part of the world starts in the index buffer, so the renderer
    /// can draw them with different pipeline state later if it needs to.
    private(set) var terrainRange: Range<Int> = 0..<0
    private(set) var roadRange: Range<Int> = 0..<0
    private(set) var buildingRange: Range<Int> = 0..<0

    private let terrain: TerrainGrid

    struct TerrainGrid {
        let x0: Double, z0: Double, cell: Double
        let nx: Int, nz: Int
        let heights: [Double]

        func height(x: Double, z: Double) -> Double {
            let fx = (x - x0) / cell, fz = (z - z0) / cell
            let i = max(0, min(nx - 2, Int(floor(fx))))
            let j = max(0, min(nz - 2, Int(floor(fz))))
            let sx = max(0, min(1, fx - Double(i))), sz = max(0, min(1, fz - Double(j)))
            let h00 = heights[j * nx + i], h10 = heights[j * nx + i + 1]
            let h01 = heights[(j + 1) * nx + i], h11 = heights[(j + 1) * nx + i + 1]
            return h00 * (1 - sx) * (1 - sz) + h10 * sx * (1 - sz)
                 + h01 * (1 - sx) * sz + h11 * sx * sz
        }

        var minX: Double { x0 }
        var maxX: Double { x0 + Double(nx - 1) * cell }
        var minZ: Double { z0 }
        var maxZ: Double { z0 + Double(nz - 1) * cell }
    }

    init(worldURL: URL, buildingsURL: URL) throws {
        let world = try JSONDecoder().decode(WorldJSON.self,
                                             from: Data(contentsOf: worldURL))
        terrain = TerrainGrid(x0: world.terrain.x0, z0: world.terrain.z0,
                              cell: world.terrain.cell, nx: world.terrain.nx,
                              nz: world.terrain.nz, heights: world.terrain.heights)

        var buildings: [BuildingsJSON.Building] = []
        if let data = try? Data(contentsOf: buildingsURL) {
            buildings = try JSONDecoder().decode(BuildingsJSON.self, from: data).buildings
        }

        let urban = UrbanMask(edges: world.edges, buildings: buildings, terrain: terrain)
        buildTerrain(urban: urban)
        buildRoads(world.edges)
        buildBuildings(buildings)
    }

    /// How built-up a patch of ground is, 0 to 1.
    ///
    /// Without this the ground between the roads renders as grass, and central
    /// Zurich comes out looking like a village on a lawn. What is actually
    /// between the buildings is pavement, courtyard and tram bed — so stamp
    /// roads and building footprints into a coarse raster and tint by it.
    struct UrbanMask {
        let x0: Double, z0: Double, cell: Double
        let nx: Int, nz: Int
        var values: [Float]

        init(edges: [WorldJSON.Edge], buildings: [BuildingsJSON.Building],
             terrain: TerrainGrid) {
            cell = 12
            x0 = terrain.minX; z0 = terrain.minZ
            nx = max(2, Int((terrain.maxX - terrain.minX) / cell) + 1)
            nz = max(2, Int((terrain.maxZ - terrain.minZ) / cell) + 1)
            values = [Float](repeating: 0, count: nx * nz)

            func stamp(_ x: Double, _ z: Double, radius: Double, weight: Float) {
                // Clamp the footprint into the grid rather than clamping a centre
                // index: buildings and roads can sit just outside the terrain
                // bounds, and Int() truncates toward zero, so a naive centre-plus-
                // radius range traps with lowerBound > upperBound.
                let i0 = max(0, Int(floor((x - x0 - radius) / cell)))
                let i1 = min(nx - 1, Int(floor((x - x0 + radius) / cell)))
                let j0 = max(0, Int(floor((z - z0 - radius) / cell)))
                let j1 = min(nz - 1, Int(floor((z - z0 + radius) / cell)))
                guard i0 <= i1, j0 <= j1 else { return }
                for j in j0...j1 {
                    for i in i0...i1 {
                        let dx = (Double(i) + 0.5) * cell + x0 - x
                        let dz = (Double(j) + 0.5) * cell + z0 - z
                        if dx * dx + dz * dz <= radius * radius {
                            values[j * nx + i] = max(values[j * nx + i], weight)
                        }
                    }
                }
            }

            for edge in edges {
                let pts = edge.p.filter { $0.count == 3 }
                guard pts.count >= 2 else { continue }
                // Wider streets sit in wider aprons of hard standing.
                let radius = max(16.0, edge.w * 2.2)
                for k in 0..<(pts.count - 1) {
                    let ax = pts[k][0], az = pts[k][2]
                    let bx = pts[k + 1][0], bz = pts[k + 1][2]
                    let len = (bx - ax).magnitude + (bz - az).magnitude
                    let steps = max(1, Int(len / 8))
                    for s in 0...steps {
                        let t = Double(s) / Double(steps)
                        stamp(ax + (bx - ax) * t, az + (bz - az) * t,
                              radius: radius, weight: 1)
                    }
                }
            }
            for b in buildings {
                let cx = b.r.reduce(0.0) { $0 + $1[0] } / Double(b.r.count)
                let cz = b.r.reduce(0.0) { $0 + $1[1] } / Double(b.r.count)
                stamp(cx, cz, radius: 22, weight: 1)
            }

            // One blur pass, so the city does not end at a hard line.
            var blurred = values
            for j in 1..<(nz - 1) {
                for i in 1..<(nx - 1) {
                    var sum: Float = 0
                    for dj in -1...1 { for di in -1...1 {
                        sum += values[(j + dj) * nx + (i + di)]
                    } }
                    blurred[j * nx + i] = sum / 9
                }
            }
            values = blurred
        }

        func sample(x: Double, z: Double) -> Float {
            let fx = (x - x0) / cell, fz = (z - z0) / cell
            let i = max(0, min(nx - 1, Int(floor(fx)))), j = max(0, min(nz - 1, Int(floor(fz))))
            return values[j * nx + i]
        }
    }

    // MARK: - Terrain

    private func buildTerrain(urban: UrbanMask) {
        let start = indices.count
        // Subdivide the 80 m sample grid: the samples carry the shape, but 80 m
        // quads give visibly faceted shading on the slopes up the Zurichberg.
        let sub = 4
        let nx = (terrain.nx - 1) * sub + 1
        let nz = (terrain.nz - 1) * sub + 1
        let step = terrain.cell / Double(sub)
        let base = UInt32(vertices.count)

        for j in 0..<nz {
            for i in 0..<nx {
                let x = terrain.x0 + Double(i) * step
                let z = terrain.z0 + Double(j) * step
                let y = terrain.height(x: x, z: z)
                let hx = terrain.height(x: x + step, z: z) - terrain.height(x: x - step, z: z)
                let hz = terrain.height(x: x, z: z + step) - terrain.height(x: x, z: z - step)
                let n = normalize(SIMD3<Float>(Float(-hx), Float(2 * step), Float(-hz)))
                // Green on the hillsides and parks, pavement grey wherever the
                // city actually is, and bare rock tint on anything steep.
                let green = SIMD3<Float>(0.28, 0.35, 0.22)
                let pavement = SIMD3<Float>(0.35, 0.34, 0.33)
                let rock = SIMD3<Float>(0.40, 0.38, 0.35)
                var colour = mix(green, pavement, t: urban.sample(x: x, z: z))
                let steep = Float(max(0, min(1, (0.97 - Double(n.y)) * 12)))
                colour = mix(colour, rock, t: steep)
                vertices.append(Vertex(position: SIMD3(Float(x), Float(y), Float(z)),
                                       normal: n, colour: colour, material: .ortho))
            }
        }
        for j in 0..<(nz - 1) {
            for i in 0..<(nx - 1) {
                let a = base + UInt32(j * nx + i)
                let b = a + 1
                let c = a + UInt32(nx)
                let d = c + 1
                indices += [a, c, b, b, c, d]
            }
        }
        terrainRange = start..<indices.count
    }

    // MARK: - Roads

    private func buildRoads(_ edges: [WorldJSON.Edge]) {
        let start = indices.count
        for edge in edges {
            let pts = edge.p.compactMap { p -> SIMD3<Double>? in
                p.count == 3 ? SIMD3(p[0], p[1], p[2]) : nil
            }
            guard pts.count >= 2 else { continue }
            let half = edge.w / 2
            let colour = roadColour(edge.c)

            for k in 0..<(pts.count - 1) {
                let a = pts[k], b = pts[k + 1]
                var dir = b - a
                dir.y = 0
                let len = simd_length(dir)
                guard len > 1e-6 else { continue }
                dir /= len
                let side = SIMD3<Double>(-dir.z, 0, dir.x) * half
                // Lift the carriageway a few centimetres. The road heights come
                // from the same terrain grid the ground mesh does, so without
                // this they are coplanar and z-fight into a shimmering mess.
                let lift = SIMD3<Double>(0, 0.08, 0)
                let quad = [a - side + lift, a + side + lift,
                            b - side + lift, b + side + lift]
                let base = UInt32(vertices.count)
                for corner in quad {
                    vertices.append(Vertex(
                        position: SIMD3(Float(corner.x), Float(corner.y), Float(corner.z)),
                        normal: SIMD3(0, 1, 0),
                        colour: colour))
                }
                indices += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
            }
        }
        roadRange = start..<indices.count
    }

    private func roadColour(_ cls: String) -> SIMD3<Float> {
        switch cls.replacingOccurrences(of: "_link", with: "") {
        // Real asphalt is dark, but a dark albedo under canyon shade crushes to
        // black once tone mapping is applied. These are lifted deliberately.
        case "motorway", "trunk":       return SIMD3(0.170, 0.170, 0.184)
        case "primary", "secondary":    return SIMD3(0.162, 0.162, 0.176)
        case "service":                 return SIMD3(0.185, 0.181, 0.176)
        default:                        return SIMD3(0.155, 0.155, 0.168)
        }
    }

    // MARK: - Buildings

    private func buildBuildings(_ list: [BuildingsJSON.Building]) {
        let start = indices.count
        for (n, b) in list.enumerated() {
            let ring = b.r.map { SIMD2<Double>($0[0], $0[1]) }
            guard ring.count >= 3 else { continue }
            let base = b.b
            let top = b.b + b.h

            // Zurich's centre is render, sandstone, pale ochre and painted
            // stucco. Vary per building or the city turns into one continuous
            // grey extrusion.
            let hash = Float((n &* 2654435761) % 997) / 997.0
            let hash2 = Float((n &* 40503) % 811) / 811.0
            let palette: [SIMD3<Float>] = [
                SIMD3(0.78, 0.75, 0.68),   // warm render
                SIMD3(0.70, 0.70, 0.67),   // grey render
                SIMD3(0.82, 0.78, 0.70),   // pale ochre
                SIMD3(0.74, 0.72, 0.68),   // stone
                SIMD3(0.66, 0.67, 0.64),   // grey-green
                SIMD3(0.85, 0.83, 0.79),   // near-white stucco
            ]
            let wall = palette[Int(hash * Float(palette.count)) % palette.count]
                       * (0.92 + hash2 * 0.14)
            let roof = mix(SIMD3<Float>(0.30, 0.26, 0.25),
                           SIMD3<Float>(0.42, 0.31, 0.27), t: hash)

            // Real storey heights are not uniform. 2.9-3.6 m spans the range
            // from a tight residential floor to a generous commercial one.
            let storey = 2.9 + hash2 * 0.7
            let params = SIMD4<Float>(Float(base), storey, hash, Float(b.h))

            // Walls
            for k in 0..<ring.count {
                let p0 = ring[k], p1 = ring[(k + 1) % ring.count]
                var edge = SIMD2<Double>(p1.x - p0.x, p1.y - p0.y)
                let len = simd_length(edge)
                guard len > 1e-6 else { continue }
                edge /= len
                let nrm = SIMD3<Float>(Float(edge.y), 0, Float(-edge.x))
                let idx = UInt32(vertices.count)
                for (p, y) in [(p0, base), (p1, base), (p0, top), (p1, top)] {
                    // Baked contact shading: a street canyon is measurably darker
                    // at pavement level, and without it the extrusions read as
                    // flat cardboard cutouts standing on the ground.
                    let occlusion: Float = (y == base) ? 0.62 : 1.0
                    vertices.append(Vertex(
                        position: SIMD3(Float(p.x), Float(y), Float(p.y)),
                        normal: nrm, colour: wall * occlusion, material: .wall,
                        params: params))
                }
                indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
            }

            // Roof
            let roofBase = UInt32(vertices.count)
            for p in ring {
                vertices.append(Vertex(
                    position: SIMD3(Float(p.x), Float(top), Float(p.y)),
                    normal: SIMD3(0, 1, 0), colour: roof, material: .roof))
            }
            for t in stride(from: 0, to: b.t.count - 2, by: 3) {
                indices += [roofBase + UInt32(b.t[t]),
                            roofBase + UInt32(b.t[t + 1]),
                            roofBase + UInt32(b.t[t + 2])]
            }
        }
        buildingRange = start..<indices.count
    }

    func terrainHeight(x: Double, z: Double) -> Double { terrain.height(x: x, z: z) }
}

private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, t: Float) -> SIMD3<Float> {
    a + (b - a) * t
}

// MARK: - On-disk

struct WorldJSON: Decodable {
    struct Terrain: Decodable {
        let x0: Double, z0: Double, cell: Double
        let nx: Int, nz: Int, heights: [Double]
    }
    struct Edge: Decodable {
        let p: [[Double]]; let w: Double; let n: String
        let c: String; let s: Int; let o: Bool; let b: Bool; let t: Bool
    }
    let name: String
    let terrain: Terrain
    let edges: [Edge]
}

struct BuildingsJSON: Decodable {
    struct Building: Decodable {
        let r: [[Double]]; let b: Double; let h: Double; let t: [Int]
    }
    let buildings: [Building]
}

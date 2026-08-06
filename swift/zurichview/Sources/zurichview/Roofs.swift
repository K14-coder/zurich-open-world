import Foundation
import simd

/// Pitched roofs.
///
/// A flat-topped extrusion is the single biggest tell that a city is generated,
/// and Zurich is a city of steep pitched roofs — from any elevated view the flat
/// tops read as a diagram no matter how good the façades or the lighting are.
///
/// The proper construction is a straight skeleton, which handles any footprint
/// but is fiddly and slow. This insets the footprint along its angle bisectors
/// and lifts the inset ring, which is the same thing for the convex-ish blocks
/// that make up most of a European street, and degrades to a flat roof rather
/// than to garbage when the footprint is too awkward.
extension WorldMesh {

    func addRoof(ring: [SIMD2<Double>], top: Double, colour: SIMD3<Float>,
                 cap: [Int], seed: Float) {
        let shortest = shortestSpan(ring)
        // Steep, as Zurich's are, but never so tall it looks like a spire.
        let pitch = 0.42 + Double(seed) * 0.22
        let rise = min(6.5, max(1.4, shortest * pitch))
        let inset = min(shortest * 0.42, rise / 1.15)

        if inset > 0.35, let lifted = insetRing(ring, by: inset) {
            addPitched(ring: ring, inner: lifted, base: top, rise: rise,
                       colour: colour, cap: cap)
        } else {
            addFlat(ring: ring, top: top, colour: colour, cap: cap)
        }
    }

    // MARK: - Flat fallback

    private func addFlat(ring: [SIMD2<Double>], top: Double,
                         colour: SIMD3<Float>, cap: [Int]) {
        let base = UInt32(vertices.count)
        for p in ring {
            vertices.append(Vertex(
                position: SIMD3(Float(p.x), Float(top), Float(p.y)),
                normal: SIMD3(0, 1, 0), colour: colour, material: .roof))
        }
        for t in stride(from: 0, to: cap.count - 2, by: 3) {
            indices += [base + UInt32(cap[t]), base + UInt32(cap[t + 1]),
                        base + UInt32(cap[t + 2])]
        }
    }

    // MARK: - Pitched

    private func addPitched(ring: [SIMD2<Double>], inner: [SIMD2<Double>],
                            base: Double, rise: Double,
                            colour: SIMD3<Float>, cap: [Int]) {
        let ridge = base + rise

        // Sloped faces: eave edge out at the footprint, ridge edge in and up.
        for i in 0..<ring.count {
            let j = (i + 1) % ring.count
            let a = ring[i], b = ring[j]
            let c = inner[j], d = inner[i]

            let p0 = SIMD3<Double>(a.x, base, a.y)
            let p1 = SIMD3<Double>(b.x, base, b.y)
            let p2 = SIMD3<Double>(c.x, ridge, c.y)
            let p3 = SIMD3<Double>(d.x, ridge, d.y)

            var n = simd_cross(p1 - p0, p3 - p0)
            let len = simd_length(n)
            guard len > 1e-9 else { continue }
            n /= len
            if n.y < 0 { n = -n }        // roof faces the sky, whatever the winding
            let nrm = SIMD3<Float>(Float(n.x), Float(n.y), Float(n.z))

            let idx = UInt32(vertices.count)
            for p in [p0, p1, p3, p2] {
                vertices.append(Vertex(
                    position: SIMD3(Float(p.x), Float(p.y), Float(p.z)),
                    normal: nrm, colour: colour, material: .roof))
            }
            indices += [idx, idx + 2, idx + 1, idx + 1, idx + 2, idx + 3]
        }

        // The flat top between the ridges. Reusing the footprint's own
        // triangulation is safe because the inset ring has the same vertex
        // count and winding — an inset never reorders anything.
        let capBase = UInt32(vertices.count)
        for p in inner {
            vertices.append(Vertex(
                position: SIMD3(Float(p.x), Float(ridge), Float(p.y)),
                normal: SIMD3(0, 1, 0), colour: colour, material: .roof))
        }
        for t in stride(from: 0, to: cap.count - 2, by: 3) {
            indices += [capBase + UInt32(cap[t]), capBase + UInt32(cap[t + 1]),
                        capBase + UInt32(cap[t + 2])]
        }
    }

    /// Small uniform shrink of a footprint, used to keep shared party walls
    /// from occupying the same plane. Returns nil rather than a folded polygon.
    func shrinkRing(_ ring: [SIMD2<Double>], by amount: Double) -> [SIMD2<Double>]? {
        insetRing(ring, by: amount)
    }

    // MARK: - Geometry

    /// Shortest distance across the footprint, which sets how tall a roof can be
    /// before it looks absurd. Approximated by the shortest edge of the bounding
    /// box, which is cheap and close enough for city blocks.
    private func shortestSpan(_ ring: [SIMD2<Double>]) -> Double {
        var minX = Double.infinity, maxX = -Double.infinity
        var minZ = Double.infinity, maxZ = -Double.infinity
        for p in ring {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minZ = min(minZ, p.y); maxZ = max(maxZ, p.y)
        }
        return min(maxX - minX, maxZ - minZ)
    }

    /// Footprint pushed inwards along its angle bisectors.
    ///
    /// Returns nil when the result folds in on itself, which happens on thin or
    /// strongly concave footprints. A flat roof there is far better than a
    /// tangle of inverted triangles.
    fileprivate func insetRing(_ ring: [SIMD2<Double>], by inset: Double) -> [SIMD2<Double>]? {
        let n = ring.count
        guard n >= 3 else { return nil }
        let ccw = signedArea(ring) > 0
        var out: [SIMD2<Double>] = []
        out.reserveCapacity(n)

        for i in 0..<n {
            let prev = ring[(i - 1 + n) % n]
            let cur = ring[i]
            let next = ring[(i + 1) % n]

            var e0 = cur - prev
            var e1 = next - cur
            let l0 = simd_length(e0), l1 = simd_length(e1)
            guard l0 > 1e-9, l1 > 1e-9 else { return nil }
            e0 /= l0; e1 /= l1

            // Inward normals of the two edges meeting at this vertex.
            let n0 = ccw ? SIMD2(-e0.y, e0.x) : SIMD2(e0.y, -e0.x)
            let n1 = ccw ? SIMD2(-e1.y, e1.x) : SIMD2(e1.y, -e1.x)
            var bisector = n0 + n1
            let bl = simd_length(bisector)
            guard bl > 1e-6 else { return nil }     // 180° spike
            bisector /= bl

            // Move far enough along the bisector that both edges shift by
            // `inset`; at a sharp corner that distance grows without bound, so
            // cap it and let the area check below reject the result.
            // At a sharp corner the bisector distance grows without bound and
            // the offset vertex shoots far outside the building, producing roof
            // faces that slice down through the façade. 0.45 caps the
            // displacement at about 2.2x the inset, which is the point past
            // which a flat roof is the better answer.
            let cosHalf = simd_dot(bisector, n0)
            guard cosHalf > 0.45 else { return nil }
            out.append(cur + bisector * (inset / cosHalf))
        }

        // A valid inset shrinks the polygon and keeps its orientation. Anything
        // else means the offset has folded through itself.
        let a0 = abs(signedArea(ring)), a1 = abs(signedArea(out))
        guard a1 > a0 * 0.25, a1 < a0, (signedArea(out) > 0) == ccw else { return nil }
        return out
    }

    private func signedArea(_ ring: [SIMD2<Double>]) -> Double {
        var s = 0.0
        for i in 0..<ring.count {
            let a = ring[i], b = ring[(i + 1) % ring.count]
            s += a.x * b.y - b.x * a.y
        }
        return s / 2
    }
}

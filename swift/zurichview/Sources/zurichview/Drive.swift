import AppKit
import MetalKit
import simd

/// A drivable window onto the city.
///
/// The car model here is deliberately minimal — throttle, brake, steering and
/// the ground under the wheels. The real one is `ApexSim.VehicleSim`, which
/// already does all of this properly and asks the world exactly one question,
/// `sampleGround(x:z:)`, which `RoadNetwork` answers. Swapping it in is the
/// integration step, not a rewrite.
final class DriveView: MTKView, MTKViewDelegate {
    private let renderer: Renderer
    private let network: RoadNetwork

    // Car state
    private var position = SIMD3<Double>(0, 0, 0)
    private var heading = 0.0          // radians, 0 = -Z
    private var speed = 0.0            // m/s
    private var steering = 0.0

    private var held = Set<UInt16>()
    private var lastTick = Date()
    private var onRoad = true

    // Key codes: W A S D, arrows, space, shift.
    private enum Key: UInt16 {
        case w = 13, a = 0, s = 1, d = 2
        case up = 126, down = 125, left = 123, right = 124
        case space = 49, escape = 53, r = 15
    }

    init(frame: CGRect, renderer: Renderer, network: RoadNetwork, start: String) {
        self.renderer = renderer
        self.network = network
        super.init(frame: frame, device: renderer.metalDevice)
        colorPixelFormat = .bgra8Unorm
        depthStencilPixelFormat = .depth32Float
        sampleCount = Renderer.sampleCount   // must match the pipelines
        clearColor = MTLClearColor(red: 0.76, green: 0.82, blue: 0.88, alpha: 1)
        preferredFramesPerSecond = 60
        delegate = self
        respawn(on: start)
    }

    required init(coder: NSCoder) { fatalError() }

    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) { held.insert(event.keyCode) }
    override func keyUp(with event: NSEvent) { held.remove(event.keyCode) }

    private func respawn(on street: String) {
        // Start on the panoramic run. That stretch is projectively textured from
        // real photographs, and it is the only part of the city that looks the
        // way this project is aiming for — spawning anywhere else shows the
        // procedural build and buries the point.
        if let spot = Self.plateViewpoint() {
            position = SIMD3(spot.x, 0, spot.z)
            heading = spot.yaw
        } else if let spawn = network.spawn(on: street) {
            position = spawn.position
            heading = spawn.yaw
        }
        speed = 0
    }

    /// Start of the panoramic run, facing along it.
    static func panoramaStart() -> (x: Double, z: Double, yaw: Double)? {
        guard let url = try? dataURL("panoramas.json"),
              let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(Panoramas.JSONFile.self, from: data)
        else { return nil }
        let ordered = file.panoramas.sorted { $0.index < $1.index }
        guard let first = ordered.first, ordered.count > 1,
              first.pos.count == 3 else { return nil }
        let next = ordered[1]
        let dx = next.pos[0] - first.pos[0]
        let dz = next.pos[2] - first.pos[2]
        return (first.pos[0], first.pos[2], atan2(dx, -dz))
    }

    /// A point on the road in front of a plated façade, facing it.
    static func plateViewpoint() -> (x: Double, z: Double, yaw: Double)? {
        guard let url = try? plateURL(),
              let data = try? Data(contentsOf: url),
              let atlas = try? JSONDecoder().decode(WorldMesh.AtlasJSON.self, from: data),
              !atlas.plates.isEmpty else { return nil }

        // Widest plate that is still plausibly a single building.
        let ranked = atlas.plates.compactMap { p -> (Double, WorldMesh.AtlasJSON.Plate)? in
            guard p.corners.count == 4, p.corners[0].count == 3 else { return nil }
            let a = p.corners[0], b = p.corners[1]
            let w = ((b[0] - a[0]) * (b[0] - a[0]) + (b[2] - a[2]) * (b[2] - a[2])).squareRoot()
            return (w, p)
        }.filter { $0.0 > 6 && $0.0 < 20 }.sorted { $0.0 > $1.0 }
        guard let chosen = ranked.first?.1 else { return nil }

        let a = chosen.corners[0], b = chosen.corners[1]
        let mx = (a[0] + b[0]) / 2, mz = (a[2] + b[2]) / 2
        var ex = b[0] - a[0], ez = b[2] - a[2]
        let len = (ex * ex + ez * ez).squareRoot()
        guard len > 1e-6 else { return nil }
        ex /= len; ez /= len
        let nx = -ez, nz = ex           // matches the winding Plates.swift uses
        let stand = 10.0
        let px = mx + nx * stand, pz = mz + nz * stand
        // Face back towards the wall.
        let dx = mx - px, dz = mz - pz
        return (px, pz, atan2(dx, -dz))
    }

    private static func plateURL() throws -> URL { try dataURL("facade_atlas.json") }

    private static func dataURL(_ name: String) throws -> URL {
        if let res = Bundle.main.resourceURL {
            let bundled = res.appendingPathComponent("data").appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: bundled.path) { return bundled }
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for c in ["..", "../..", "../../data", "../data", "data", "."] {
            let u = cwd.appendingPathComponent(c).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: u.path) { return u.standardized }
        }
        throw Fail("missing \(name)")
    }

    private var forward: SIMD3<Double> {
        SIMD3(sin(heading), 0, -cos(heading))
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = Date()
        let dt = min(0.05, now.timeIntervalSince(lastTick))
        lastTick = now
        step(dt: dt)

        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = renderer.commandQueue.makeCommandBuffer()
        else { return }

        let ground = network.sampleGround(x: position.x, z: position.z)
        let eye = SIMD3<Float>(Float(position.x),
                               Float(ground.height + 1.32),
                               Float(position.z))
        let dir = SIMD3<Float>(Float(forward.x), 0, Float(forward.z))
        let camera = Camera(eye: eye, target: eye + dir * 60 - SIMD3(0, 1.6, 0),
                            fovDegrees: 68)

        renderer.encode(camera: camera, pass: pass,
                        width: Int(view.drawableSize.width),
                        height: Int(view.drawableSize.height), into: cb)
        cb.present(drawable)
        cb.commit()
    }

    private func step(dt: Double) {
        let throttle = (held.contains(Key.w.rawValue) || held.contains(Key.up.rawValue)) ? 1.0 : 0.0
        let brake = (held.contains(Key.s.rawValue) || held.contains(Key.down.rawValue)) ? 1.0 : 0.0
        var steerInput = 0.0
        if held.contains(Key.a.rawValue) || held.contains(Key.left.rawValue) { steerInput -= 1 }
        if held.contains(Key.d.rawValue) || held.contains(Key.right.rawValue) { steerInput += 1 }
        if held.contains(Key.r.rawValue) { respawn(on: "Freiestrasse") }
        if held.contains(Key.escape.rawValue) { NSApp.terminate(nil) }

        let sample = network.sampleGround(x: position.x, z: position.z)
        onRoad = sample.surface == .asphalt
        // Off the tarmac you lose grip, exactly as the tire model would have it.
        let grip = sample.surface.gripMultiplier

        let drive = 7.5 * throttle * grip
        let drag = 0.011 * speed * abs(speed) + 0.55
        let braking = 12.0 * brake * grip
        speed += (drive - braking - copysign(drag, speed)) * dt
        if throttle == 0 && brake == 0 && abs(speed) < 0.4 { speed = 0 }
        speed = max(-6, min(48, speed))

        // Steering authority falls away with speed, which is what stops the car
        // spinning on the spot at 40 m/s.
        steering += (steerInput - steering) * min(1, dt * 8)
        let authority = 1.6 / (1 + abs(speed) * 0.09)
        heading += steering * authority * dt * min(1, abs(speed) / 3)

        position += forward * speed * dt
        position.y = sample.height
    }

    var hudTitle: String {
        String(format: "Zurich — %.0f km/h — %@", speed * 3.6, onRoad ? "on road" : "off road")
    }
}

func runInteractive(mesh: WorldMesh, network: RoadNetwork, renderer: Renderer) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)

    let frame = NSRect(x: 0, y: 0, width: 1280, height: 780)
    let window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable],
                          backing: .buffered, defer: false)
    window.title = "Zurich"
    window.center()

    let view = DriveView(frame: frame, renderer: renderer, network: network,
                         start: "Freiestrasse")
    window.contentView = view
    window.makeFirstResponder(view)
    window.makeKeyAndOrderFront(nil)

    Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { _ in
        window.title = view.hudTitle
    }

    app.activate(ignoringOtherApps: true)
    app.run()
}

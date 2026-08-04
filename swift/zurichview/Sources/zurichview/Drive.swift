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
        if let spawn = network.spawn(on: street) {
            position = spawn.position
            heading = spawn.yaw
        }
        speed = 0
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

// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

print("Hello, World!")

let netDispatch = DispatchQueue(
    label: "cc.dispatch.net", target: DispatchQueue.global()
)

let callbackDispatch = DispatchQueue(
    label: "cb.dispatch.net", target: DispatchQueue.global()
)

let semaphore = DispatchSemaphore(value: 0)
let semaphore2 = DispatchSemaphore(value: 0)

class Barf {
    var r0: SwiftPointer<Float>?
    var r1: UnsafePointer<Float>?
    var r2: UnsafeMutablePointer<Float>?

    init() {
        r0 = SwiftPointer(Float.self, bytes: 100)
        r1 = r0!.asImmutable()
        r2 = r0!.asMutable()
    }
}

var barf: Barf? = Barf()
barf = nil

class Gremlin {
    var idNumber = 0

    var inputLayer = SwiftnetLayer(
        activation: .identity, poolingFunction: .max,
        kernelWidth: 3, height: 3, cChannels: 2
    )

    var hiddenLayers = [SwiftnetLayer(activation: .identity, cInputs: 18, cOutputs: 9, cChannels: 1)]

    var outputLayer = SwiftnetLayer(
        activation: .identity, cInputs: 9, cOutputs: 9, cChannels: 1, calculateControls: !lastLayerIsEmpty
    )

    var allLayers: [SwiftnetLayer] { [inputLayer] + hiddenLayers }// + [outputLayer] }

    var net: Swiftnet!
    var pBiases: UnsafeMutableRawPointer!
    var pIO: UnsafeMutableRawPointer!
    var pWeights: UnsafeMutableRawPointer!

    init(idNumber: Int) {
        self.idNumber = idNumber
        let counts = Swiftnet.getStackCounts(allLayers)

        self.pBiases = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cBiases), alignment: MemoryLayout<Float>.alignment)
        self.pIO = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cInputs + counts.cOutputs), alignment: MemoryLayout<Float>.alignment)
        self.pWeights = UnsafeMutableRawPointer.allocate(byteCount: bytes(counts.cWeights), alignment: MemoryLayout<Float>.alignment)

        pBiases.initializeMemory(as: Float.self, repeating: 0, count: counts.cBiases)
        pWeights.initializeMemory(as: Float.self, repeating: 1, count: counts.cWeights)

        SwiftPointer<Float>.toMutableBuffer(from: pIO, cElements: counts.cInputs).initialize(repeating: Float(idNumber + 1))

        self.net = Swiftnet(
            counts: counts, layers: allLayers,
            pBiases: pBiases, pIO: pIO, pWeights: pWeights
        )
    }
}

let count = 1

func expand(_ number: Int) -> [Float] {
    var n = number
    var f = [Float](repeating: 0, count: 9)

    for i in 0..<9 {
        if ((n & 1) != 0) { f[i] = 1 }
        n >>= 1
    }

    return f
}

func testInputs(_ net: Swiftnet) {
    for i in 0..<Int(pow(2.0, 9.0)) {
        print(expand(i))
    }
}

for idNumber in 0..<count {
    netDispatch.async {
        let g = Gremlin(idNumber: idNumber)

        let h = UnsafeMutableBufferPointer(mutating: g.net.inputBuffer)

        let ii = g.net.pInputs.assumingMemoryBound(to: Float.self)
        let i = UnsafeMutableBufferPointer<Float>(start: ii + 9, count: 9)

        for ix in 0..<Int(pow(2.0, 9.0)) {
            let f = expand(ix)
            h.indices.forEach { h[$0] = f[$0] }
            g.net.activate()
            print("in \(f.reversed().map { $0 }) -> mid \(i.map { $0 }) -> out \(g.net.outputBuffer.map { $0 })")
        }

        semaphore2.signal()
    }
}

for _ in 0..<count { semaphore2.wait() }

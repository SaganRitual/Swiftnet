// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

typealias Octuple = (Int, Int, Int, Int, Int, Int, Int, Int)
let OctupleZero = (0, 0, 0, 0, 0, 0, 0, 0)

func * (_ lhs: Octuple, _ rhs: Octuple) -> Octuple {
    (lhs.0 * rhs.0, lhs.1 * rhs.1, lhs.2 * rhs.2, lhs.3 * rhs.3,
     lhs.4 * rhs.4, lhs.5 * rhs.5, lhs.6 * rhs.6, lhs.7 * rhs.7)
}

enum SwiftnetFilter {
    static func makeFilter(
        _ layer: SwiftnetLayer,
        _ pBiases: UnsafeMutableRawPointer?,
        _ pInputs: UnsafeMutableRawPointer,
        _ pOutputs: UnsafeMutableRawPointer,
        _ pWeights: UnsafeMutableRawPointer?
    ) -> BNNSFilter {
        switch layer.layerType {
        case .fullyConnected:
            return SwiftnetFilter.makeFullyConnectedFilter(
                layer, pBiases, pInputs, pOutputs, pWeights
            )

        case .pooling:
            return SwiftnetFilter.makePoolingFilter(
                layer, pBiases, pInputs, pOutputs
            )
        }
    }

    static var bnnsFilterParameters = BNNSFilterParameters(
        flags: BNNSFlags.useClientPointer.rawValue, n_threads: 0,
        alloc_memory: nil, free_memory: nil
    )

    static func makeArrayDescriptor(
        layout: BNNSDataLayout = BNNSDataLayoutVector,
        size: Octuple = OctupleZero,
        data: UnsafeMutableRawPointer? = nil
    ) -> BNNSNDArrayDescriptor {
        .init(
            flags: BNNSNDArrayFlags(0),
            layout: layout,
            size: size,
            stride: OctupleZero,
            data: data,
            data_type: .float,
            table_data: nil,
            table_data_type: .float,
            data_scale: 0,
            data_bias: 0
        )
    }

    static func makeFullyConnectedFilter(
        _ layer: SwiftnetLayer,
        _ pBiases: UnsafeMutableRawPointer?,
        _ pInputs: UnsafeMutableRawPointer,
        _ pOutputs: UnsafeMutableRawPointer,
        _ pWeights: UnsafeMutableRawPointer?
    ) -> BNNSFilter {
        assert(
            layer.counts.cBiases > 0 || pBiases == nil,
            "Zero biases this layer, but non-nil pointer to biases buffer"
        )

        assert(
            layer.counts.cWeights > 0 || pWeights == nil,
            "Zero weights this layer, but non-nil pointer to weights buffer"
        )

        assert(
            layer.counts.cInputs > 0 && layer.counts.cOutputs > 0,
            "cInputs \(layer.counts.cInputs) cOutputs \(layer.counts.cOutputs), not good"
        )

        let biases = layer.counts.cBiases == 0 ?

            SwiftnetFilter.makeArrayDescriptor() :

            SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutVector,
            size: (layer.counts.cBiases, 1, 0, 0, 0, 0, 0, 0),
            data: pBiases
        )

        let inputs = SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutVector,
            size: (layer.counts.cInputs, 1, 0, 0, 0, 0, 0, 0),
            data: nil
        )

        let outputs = SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutVector,
            size: (layer.counts.cOutputs, 1, 0, 0, 0, 0, 0, 0),
            data: nil
        )

        let weights = layer.counts.cWeights == 0 ?

            SwiftnetFilter.makeArrayDescriptor() :

            SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutRowMajorMatrix,
            size: (layer.counts.cInputs, layer.counts.cOutputs, 0, 0, 0, 0, 0, 0),
            data: pWeights
        )

        var lp = BNNSLayerParametersFullyConnected(
            i_desc: inputs, w_desc: weights, o_desc: outputs,
            bias: biases, activation: BNNSActivation(function: layer.activation)
        )

        return BNNSFilterCreateLayerFullyConnected(&lp, &bnnsFilterParameters)!
    }

    static func makePoolingFilter(
        _ layer: SwiftnetLayer,
        _ pBiases: UnsafeMutableRawPointer?,
        _ pInputs: UnsafeMutableRawPointer,
        _ pOutputs: UnsafeMutableRawPointer
    ) -> BNNSFilter {
        let inputs = SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutImageCHW,
            size: (layer.width, layer.height, layer.cChannels, 0, 0, 0, 0, 0)
        )

        let outputs = SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutImageCHW,
            size: (layer.width, layer.height, 1, 0, 0, 0, 0, 0)
        )

        let biases = SwiftnetFilter.makeArrayDescriptor(
            layout: BNNSDataLayoutVector,
            size: (layer.width * layer.height, 0, 0, 0, 0, 0, 0, 0),
            data: pBiases
        )

        var lp = BNNSLayerParametersPooling(
            i_desc: inputs, o_desc: outputs, bias: biases,
            activation: BNNSActivation(function: layer.activation),
            pooling_function: layer.poolingFunction,
            k_width: layer.width, k_height: layer.height,
            x_stride: 2, y_stride: 1,
            x_dilation_stride: 0, y_dilation_stride: 0,
            x_padding: layer.width / 2, y_padding: layer.height / 2,
            pad: (0, 0, 0, 0)
        )

        return BNNSFilterCreateLayerPooling(&lp, &bnnsFilterParameters)!
    }
}

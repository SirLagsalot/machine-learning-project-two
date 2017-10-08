import java.util.List;

public class RadialBasisNetwork extends NeuralNetwork {

    private int gaussiasnFunctions;
    private IActivationFunction activationFunction;

    public RadialBasisNetwork(int inputs, int outputs, int gaussianFunctions, IActivationFunction activationFunction) {
        super(inputs, outputs);
        this.gaussiasnFunctions = gaussianFunctions;
        this.activationFunction = activationFunction;
    }

    @Override
    public void train(List<Sample> samples) {

    }

    @Override
    public double[] approximate(double[] inputs) {
        return new double[inputs.length];
    }
}

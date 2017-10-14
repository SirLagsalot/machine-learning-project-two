import java.util.List;

public abstract class NeuralNetwork implements IFunctionApproximator {

    protected IActivationFunction activationFunction;

    protected int numInputs;
    protected int numOutputs;

    public NeuralNetwork(int numInputs, int numOutputs, IActivationFunction activationFunction) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    public abstract void train(List<Sample> samples);

    public abstract double[] approximate(double[] inputs);

    protected double[] computeOutputErrorGradient(double[] networkOutputs, double[] expectedOutputs) {
        double[] error = new double[networkOutputs.length];
        for (int i = 0; i < networkOutputs.length; i++) {
            error[i] = (networkOutputs[i] - expectedOutputs[i]) * this.activationFunction.computeDerivative(networkOutputs[i]);
        }
        return error;
    }
}

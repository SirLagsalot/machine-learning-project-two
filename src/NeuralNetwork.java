import java.util.List;

public abstract class NeuralNetwork implements IFunctionApproximator {

    protected IActivationFunction activationFunction;

    protected int numInputs;
    protected int numOutputs;

    public NeuralNetwork(int numInputs, int numOutputs) {
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

    protected double calculateTotalError(double[] networkOutputs, double[] expectedOutputs) {
        assert networkOutputs.length == expectedOutputs.length;

        double errorSum = 0.0;
        // Calculate the sum over the squared error for each output value
        for (int i = 0; i < networkOutputs.length; i++) {
            double error = networkOutputs[i] - expectedOutputs[i];
            errorSum += Math.pow(error, 2);
        }

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
    }
}

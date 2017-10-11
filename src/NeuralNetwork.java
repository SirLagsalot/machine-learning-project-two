import java.util.List;

public abstract class NeuralNetwork implements IFunctionApproximator {

    protected int numInputs;
    protected int numOutputs;

    public NeuralNetwork(int numInputs, int numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    public abstract void train(List<Sample> samples);

    public abstract double[] approximate(double[] inputs);
}

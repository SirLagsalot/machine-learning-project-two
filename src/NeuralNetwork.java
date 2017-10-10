import java.util.List;

public abstract class NeuralNetwork implements IFunctionApproximator {

    protected int inputs;
    protected int outputs;

    public NeuralNetwork(int inputs, int outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public abstract void train(List<Sample> samples);

    public abstract double[] approximate(double[] inputs);
}

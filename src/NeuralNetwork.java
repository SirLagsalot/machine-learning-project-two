
public abstract class NeuralNetwork implements IFunctionApproximator {

    int inputs;
    int outputs;

    public NeuralNetwork(int inputs, int outputs) {

    }

    public abstract void train();
}

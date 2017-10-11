import java.util.List;

public class RadialBasisNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private int numNeurons;
    private double[] means;

    public RadialBasisNetwork(int inputs, int outputs, int numNeurons) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.numNeurons = numNeurons;
    }

    @Override
    public void train(List<Sample> samples) {

    }

    @Override
    public double[] approximate(double[] inputs) {
        return new double[inputs.length];
    }

    //get mean of cluster
    public double getMean() {
        return 0.0;
    }
}

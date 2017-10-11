import java.util.List;

public class RadialBasisNetwork extends NeuralNetwork {

    private int gaussiasnFunctions;

    public RadialBasisNetwork(int inputs, int outputs, int gaussianFunctions) {
        super(inputs, outputs);
        this.gaussiasnFunctions = gaussianFunctions;
    }

    @Override
    public void train(List<Sample> samples) {

    }

    @Override
    public double[] approximate(double[] inputs) {
        return new double[inputs.length];
    }

    //get mean of cluster
    public double getMean(){
        return 0.0;
    }
}

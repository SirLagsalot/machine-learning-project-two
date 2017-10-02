import java.util.Random;

public class Neuron implements INeuron {

    private int numConnections;
    private double[] weights;
    private IActivationFunction activationFunction;

    public Neuron(IActivationFunction activationFunction, int numConnections) {
        this.activationFunction = activationFunction;
        this.numConnections = numConnections;
        this.weights = new double[numConnections];
        randomizeWeights(new Random());
    }

    public double propagate(double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += (weights[i] * inputs[i]);
        }

        return this.activationFunction.compute(sum);
    }

    public double[] getWeights() {
        return this.weights;
    }

    public int getNumConnections() {
        return this.numConnections;
    }

    private void randomizeWeights(Random random) {
        for (int i = 0; i < weights.length - 1; i++) {
            weights[i] = random.nextDouble();
        }
    }
}

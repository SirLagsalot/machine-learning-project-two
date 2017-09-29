import java.util.Random;

public class Neuron implements INeuron {

    private IActivationFunction activationFunction;
    private double[] weights;

    public Neuron(IActivationFunction activationFunction, int numConnections) {
        weights = new double[numConnections];
        randomizeWeights(new Random());
    }

    public int compute(double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += (weights[i] * inputs[i]);
        }

        return this.activationFunction.process(sum);
    }

    public double[] getWeights() {
        return this.weights;
    }

    private void randomizeWeights(Random random) {
        for (int i = 0; i < weights.length - 1; i++) {
            weights[i] = random.nextDouble();
        }
    }
}

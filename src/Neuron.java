import java.util.ArrayList;
import java.util.Random;

public class Neuron implements INeuron {

    private int inputs;
    private double[] weights;
    private IActivationFunction activationFunction;

    public Neuron(IActivationFunction activationFunction, int inputs) {
        this.activationFunction = activationFunction;
        this.inputs = inputs;
        this.weights = new double[inputs];
        randomizeWeights(new Random());
    }

    public double propagate(ArrayList<Double> inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.size(); i++) {
            sum += (weights[i] * inputs.get(i));
        }

        return this.activationFunction.compute(sum);
    }

    public double[] getWeights() {
        return this.weights;
    }

    public int getNumInputs() {
        return this.inputs;
    }

    private void randomizeWeights(Random random) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() * 2 - 1;   // Random b/w -1, 1
        }
    }
}

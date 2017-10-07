import java.util.ArrayList;
import java.util.Random;

public class Neuron {

    private int inputs;
    private double output;
    private double[] weights;
    private double learningRate;
    private double delta;
    private IActivationFunction activationFunction;

    public Neuron(int inputs, double learningRate, IActivationFunction activationFunction) {
        this.inputs = inputs;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        this.weights = new double[inputs];
        randomizeWeights(new Random());
    }

    public double propagate(ArrayList<Double> inputs) {
        double sum = 1; //Bias of 1 to prevent issues with 0 input

        for (int i = 0; i < inputs.size(); i++) {
            sum += (weights[i] * inputs.get(i));
        }

        return this.activationFunction.compute(sum);
    }

    public double getWeight(int index) {
        return this.weights[index];
    }

    private void randomizeWeights(Random random) {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = random.nextDouble() * 2 - 1;   // Random b/w -1, 1
        }
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getOutput() {
        return output;
    }
}

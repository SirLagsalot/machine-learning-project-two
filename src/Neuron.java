import java.util.Random;

public class Neuron {

    private double output = 0.0;
    private double bias = 1.0;
    private double delta = 0.0;
    private double[] weights;

    public Neuron(int prevLayerSize) {
        Random random = new Random();

        this.weights = new double[prevLayerSize];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = random.nextDouble();
        }
    }

    public double getWeight(int index) {
        return this.weights[index];
    }

    public void setWeight(int index, double weight) {
        this.weights[index] = weight;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}
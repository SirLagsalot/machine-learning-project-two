import java.util.Random;

public class Neuron {

    public double output = 0.0;
    public double bias = 1.0;
    public double delta = 0.0;
    public double[] weights;

    public Neuron(int prevLayerSize) {
        Random random = new Random();

        this.weights = new double[prevLayerSize];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = random.nextDouble();
        }
    }
}
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Radial basis neural network
 * Initialized with the number of inputs and outputs the network will be working with
 * along with a learning rate, batch update size, and the number of epochs the network will iterate over while learning
 */
public class RadialBasisNetwork extends NeuralNetwork {

    private double learnRate;
    private int batchSize;
    private int epochs;
    private List<Neuron> hiddenLayer;
    private List<Neuron> outputLayer;

    private final int size;

    public RadialBasisNetwork(int inputs, int outputs, int hiddenNodes, double learnRate, int batchSize, int epochs) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.size = hiddenNodes;
        this.learnRate = learnRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.initializeNetwork();
    }

    @Override
    public void train(List<Sample> samples) {
        this.initializeMeans(samples);

        double maxDist = 0.0;
        for (int i = 0; i < this.size; i++) {
            for (int j = i + 1; j < this.size; j++) {
                double dist = this.computeDistance(this.hiddenLayer.get(i).getInputs(), this.hiddenLayer.get(j).getInputs());
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
        }

        GaussianFunction.setSigma(maxDist, size);

        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            for (int j = 0; j < samples.size(); j++) {
                Sample sample = samples.get(j);
                double[] networkOutputs = this.computeGaussianActivation(sample.inputs);
                if (j % batchSize == 0) {
                    updateWeights(networkOutputs, sample.outputs);
                }

                epochError += this.calculateTotalError(sample.outputs, networkOutputs);
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        double[] outputs = this.computeGaussianActivation(inputs);
        double[] approx = new double[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            approx[i] = sumOutputs(outputs);
        }

        return approx;
    }

    private void initializeNetwork() {
        // Set up hidden layer
        this.hiddenLayer = new ArrayList<>(this.size);
        for (int i = 0; i < size; i++) {
            this.hiddenLayer.add(new Neuron(this.numInputs, false));
        }
        // Set up output layer
        this.outputLayer = new ArrayList<>(this.numOutputs);
        for (int i = 0; i < this.numOutputs; i++) {
            this.outputLayer.add(new Neuron(this.size, true));
        }
    }


    private void initializeMeans(List<Sample> samples) {
        Random random = new Random(System.nanoTime());
        for (int i = 0; i < this.size; i++) {
            int initIndex = random.nextInt(samples.size());
            Sample targetSample = samples.get(initIndex);
            this.hiddenLayer.get(i).setInputs(targetSample.inputs);
        }
    }

    // Compute the euclidean distance between the to arrays
    private double computeDistance(double[] x, double[] y) {
        assert x.length == y.length;

        double sum = IntStream
                .range(0, x.length)
                .mapToDouble(i -> Math.pow(x[i] - y[i], 2))
                .sum();

        return Math.sqrt(sum);
    }

    private void updateWeights(double[] networkOutputs, double[] expectedOutputs) {
        //update the weights of each neuron
        assert networkOutputs.length == expectedOutputs.length;

        double[] partialErrors = this.computeOutputErrorGradient(networkOutputs, expectedOutputs);
        for (int i = 0; i < this.outputLayer.size(); i++) {
            Neuron outputNeuron = this.outputLayer.get(i);
            for (int j = 0; j < this.size; j++) {
                Neuron hiddenNeuron = this.hiddenLayer.get(j);
                outputNeuron.updateWeight(j, partialErrors[i] * this.learnRate * hiddenNeuron.getOutput());
            }
        }
    }

    private double[] computeGaussianActivation(double[] inputs) {
        double[] outputs = new double[this.outputLayer.size()];

        // Loop over output neurons
        for (int i = 0; i < this.outputLayer.size(); i++) {
            Neuron outputNeuron = this.outputLayer.get(i);
            double output = 0.0;
            // Sum over connections
            for (int j = 0; j < outputNeuron.size; j++) {
                output += outputNeuron.getWeight(j) * this.computeDistance(inputs, this.hiddenLayer.get(j).getInputs());
            }
            outputs[i] = output;
        }

        return outputs;
    }

    private double sumOutputs(double[] gaussOutput) {
        return Arrays.stream(gaussOutput).sum();
    }
}

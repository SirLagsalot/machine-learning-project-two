import java.util.Arrays;
import java.util.List;

public class Tester {

    public static void main(String[] args) {
        int[] layers = new int[]{2, 200, 1};

        IFunctionApproximator neuralNetwork = new FeedForwardNetwork(2, 1, layers, 0.01, new SigmoidFunction());
        List<Sample> trainingSamples = SampleGenerator.generateSuperEasySamples(10000);
        neuralNetwork.train(trainingSamples);

        List<Sample> testSamples = SampleGenerator.generateSuperEasySamples(10);

        for (Sample sample : testSamples) {
            double networkOutput = neuralNetwork.approximate(sample.inputs)[0];
            System.out.println("Inputs: " + Arrays.toString(sample.inputs) + " Approx: " + networkOutput + " Actual: " + sample.outputs[0]);
            System.out.println("Error: " + Math.abs(networkOutput - sample.outputs[0]) + "\n");
        }
    }
}

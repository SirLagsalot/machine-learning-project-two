
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

    public static void crossValidate() {
        int numSamples = 10000;
        int inputs = 2;
        int maxInput = 5;
        int outputs = 1;
        int[] layers = {2, 50, 1};
        double learningRate = 0.01;

        List<Sample> samples = SampleGenerator.generateSamples(10000, 2, 5, 1);
        for (int k = 0; k < 5; k++) {
            List<Sample> train = samples.subList(0, (samples.size() / 2));
            List<Sample> test = samples.subList((samples.size() / 2), samples.size());

            IFunctionApproximator feed = new FeedForwardNetwork(inputs, outputs, layers, learningRate, new SigmoidFunction());
            IFunctionApproximator radial = new RadialBasisNetwork(inputs, outputs, 10);

            feed.train(train);
            // radial.train(train);

            for (int i = 0; i < test.size(); i++) {
                Sample sample = test.get(i);
                double[] networkOutput = feed.approximate(sample.inputs);
                double errorSum = 0;
                for (int j = 0; j < test.size(); j++) {
                    double error = Math.abs(networkOutput[j] - sample.outputs[j]);
                    errorSum = errorSum + error;
                }
            }
        }
    }
}

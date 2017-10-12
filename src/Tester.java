
import java.util.List;

public class Tester {

    /* Tunable Parameters */
    private static final int numInputs = 2;
    private static final int numOutputs = 1;
    private static final int maxInputVal = 3;
    private static final int numSamples = 10000;
    private static final int[] layers = new int[]{numInputs, 5, numOutputs};    // Size of each layer
    private static final int batchSize = 5;
    private static final double learningRate = 0.05;
    private static final double momentum = 0.4;                                 // Set to 0 to disable momentum

    public static void main(String[] args) {

        IFunctionApproximator neuralNetwork = new FeedForwardNetwork(layers, learningRate, batchSize, momentum, new SigmoidFunction());
        List<Sample> trainingSamples = SampleGenerator.generateSamples(numSamples, numInputs, maxInputVal, numOutputs);
        neuralNetwork.train(trainingSamples);

        List<Sample> testSamples = SampleGenerator.generateSamples(numSamples, numInputs, maxInputVal, numOutputs);

        double error = 0.0;
        for (Sample sample : testSamples) {
            double output = neuralNetwork.approximate(sample.inputs)[0];
            error += Math.abs(output - sample.outputs[0]);
        }

        System.out.println("Avg Error: " + error / numSamples);
    }

    public static void crossValidate() {
        double errorSum = 0;
        double[] totalErrorSum = new double[numSamples];

        List<Sample> samples = SampleGenerator.generateSamples(10000, 2, 5, 1);
        for (int k = 0; k < 5; k++) {
            List<Sample> train = samples.subList(0, (samples.size() / 2));
            List<Sample> test = samples.subList((samples.size() / 2), samples.size());

            IFunctionApproximator FFN = new FeedForwardNetwork(layers, learningRate, batchSize, momentum, new SigmoidFunction());
            IFunctionApproximator RBN = new RadialBasisNetwork(numInputs, numOutputs, 10);

            FFN.train(train);
            // radial.train(train);

            //iterates through test and calculates the approximated values
            for (int i = 0; i < test.size(); i++) {
                Sample sample = test.get(i); //gets element in test at i
                double[] networkOutput = FFN.approximate(sample.inputs); //gets the approximated values of the numInputs and puts them in a double array

                for (int j = 0; j < test.size(); j++) { //calculates the error for one input
                    double error = Math.abs(networkOutput[j] - sample.outputs[j]);
                    errorSum = errorSum + error; //adds the error from one input to the total error sum for the sample
                }
                totalErrorSum[i] = errorSum;
            }
            double mean = 0;
            double sum = 0;
            for (int i = 0; i < totalErrorSum.length; i++) {
                sum += totalErrorSum[i];
                mean = sum / totalErrorSum.length;
            }
            System.out.print("the mean of the Errors is: " + mean);

            double sd = 0;
            double standardDeviation = 0;
            for (int i = 0; i < totalErrorSum.length; i++) {
                sd += ((totalErrorSum[i] - mean) * (totalErrorSum[i] - mean)) / totalErrorSum.length - 1;
                standardDeviation = Math.sqrt(sd);
            }
            System.out.print("the standard deviation of the errors is: " + standardDeviation);
        }
    }
}

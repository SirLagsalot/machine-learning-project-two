import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Utility class for generating sample data from different functions
 * Primary purpose in to provide sample training and test data for the Rosenbrock function
 * Additionally provides samples for similar functions such as sine and quadratic
 */
public class SampleGenerator {

    // Generate a list of samples based on the number of inputs to the rosenbrock
    // Number of samples generated is a function of the input dimensions
    public static List<Sample> generateSamples(int numInputs) {
        int numSamples = (int) Math.pow(numInputs, 3) * 5000;
        int minValue = -3;
        int maxValue = 3;
        double gridWidth = ((double) (maxValue - minValue)) / numSamples * numInputs;

        ArrayList<Double> inputValues = new ArrayList<>();
        for (double i = (double) minValue; i < maxValue; i += gridWidth) {
            inputValues.add(i);
        }

        List<Sample> samples = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            Random random = new Random(System.currentTimeMillis());
            double[] inputs = new double[numInputs];
            for (int j = 0; j < numInputs; j++) {
                inputs[j] = inputValues.get(random.nextInt(inputValues.size()));
            }

            double[] output = new double[]{computeRosenbrockOutput(inputs)};
            samples.add(new Sample(inputs, output));
        }

        return samples;
    }

    // Generate rosenbrock samples using more specific parameters
    public static List<Sample> generateSamples(int numSamples, int numInputs, int maxInputVal, int numOutputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs, maxInputVal, numOutputs);
            sample.outputs[0] = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    // Compute the output of the Rosenbrock function given its inputs
    public static double computeRosenbrockOutput(double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2);
        }

        return sum;
    }

    // Generate samples for a variation of the quadratic formula for leaning purposes
    public static List<Sample> generateQuadraticSamples(int numSamples) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(3);
            double a = random.nextDouble() * 10 - 5;
            double b = random.nextDouble() * 10 - 5;
            double c = random.nextDouble() * 10 - 5;

            sample.inputs = new double[]{a, b, c};
            sample.outputs = new double[]{(-b + Math.sqrt(Math.abs(Math.pow(b, 2) - 4 * a * c))) / (2 * a)};
            samples.add(sample);
        }

        return samples;
    }

    // One input [-3, 3], one output sin(input)
    public static List<Sample> generateSinSamples(int numSamples) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            double[] input = new double[]{random.nextDouble() * 10 - 5};
            double[] output = new double[]{Math.sin(input[0])};
            samples.add(new Sample(input, output));
        }

        return samples;
    }
}

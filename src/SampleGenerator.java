import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SampleGenerator {

    public static List<Sample> generateSamples(int numSamples, int numInputs, int maxInputVal, int numOutputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs, maxInputVal, numOutputs);
            sample.outputs[0] = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    public static double computeRosenbrockOutput(double[] inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.length - 1; i++) {
            sum += Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2);
        }

        return sum;
    }

    public static List<Sample> generateQuadraticSamples() {
        int numSamples = 10000;
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(3);
            int a = random.nextInt(10);
            int b = random.nextInt(10);
            int c = random.nextInt(10);

            sample.inputs = new double[]{a, b, c};
            sample.outputs = new double[]{(-b + Math.sqrt(Math.abs(Math.pow(b, 2) - 4 * a * c))) / (2 * a)};
            samples.add(sample);
        }

        return samples;
    }


    public static List<Sample> generateEasySamples(int numSamples, int numInputs) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs);
            for (int j = 0; j < numInputs; j++) {
                sample.inputs[j] = (double) random.nextInt(5);
            }
            sample.outputs[0] = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    // Samples for cos(a * b) where a, b are doubles [0, 1]
    public static List<Sample> generateSuperEasySamples(int numSamples) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(2);
            for (int j = 0; j < 2; j++) {
                sample.inputs[j] = random.nextDouble();
            }
            sample.outputs[0] = Math.cos(sample.inputs[0] * sample.inputs[1]);
            samples.add(sample);
        }

        return samples;
    }
}

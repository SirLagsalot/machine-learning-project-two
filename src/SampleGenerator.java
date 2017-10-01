import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class SampleGenerator {

    public static List<Sample> generateSamples(int numSamples, int numInputs, int maxInputVal) {
        ArrayList<Sample> samples = new ArrayList<>(numSamples);
        Random random = new Random();

        for (int i = 0; i < numSamples; i++) {
            Sample sample = new Sample(numInputs, maxInputVal);
            sample.output = computeRosenbrockOutput(sample.inputs);
            samples.add(sample);
        }

        return samples;
    }

    private static double computeRosenbrockOutput(double[] inputs) {
        int n = inputs.length;
        return IntStream
                .range(0, n - 2)
                .mapToDouble(i -> Math.pow(1 - inputs[i], 2) + 100 * Math.pow(inputs[i + 1] - Math.pow(inputs[i], 2), 2))
                .sum();
    }
}

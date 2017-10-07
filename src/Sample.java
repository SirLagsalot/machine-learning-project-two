import java.util.Random;

public class Sample {
    public Double[] inputs;
    public Double[] outputs;

    public Sample(int numInputs, int maxInputVal) {
        this.inputs = new Double[numInputs];

        Random random = new Random();
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = random.nextDouble() * maxInputVal;
        }
    }
}

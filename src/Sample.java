import java.util.Random;

public class Sample {
    public Double[] inputs;
    public Double[] outputs;

    public Sample(int numInputs, int maxInputVal, int numOutputs) {
        this.inputs = new Double[numInputs];
        this.outputs = new Double[numOutputs];

        Random random = new Random();
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = random.nextDouble() * maxInputVal;
        }
    }
}

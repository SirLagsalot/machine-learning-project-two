import java.util.List;
import java.util.Random;

public class RadialBasisNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private int numNeurons;
    private Sample[] means; // store a mean for each cluster
    private double learnRate;
    private double[] errors; // average errors from each gradient descent iteration
    private int batchSize;
    private int iterations = 1000;
    private Layer layer;

    public RadialBasisNetwork(int inputs, int outputs, int numNeurons, double learnRate, int batchsize) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.numNeurons = numNeurons;
        this.learnRate = learnRate;
        this.batchSize = batchsize;
        means = new Sample[numNeurons];

    }

    @Override
    public void train(List<Sample> samples) {
        setMeans(samples);
        layer = new Layer(numNeurons, 1);
        for (int i = 0; i < layer.size; i++){
            layer.getNeuron(i).setOutput(means[i].inputs[0]); //using outputs to store the input to compare with the
        }

        double maxDist = distance(layer.getNeuron(0).getOutput(), layer.getNeuron(1).getOutput());

        for(int i = 0; i < layer.size; i ++){
            for(int j = i + 1; j < layer.size; j++){
                double dist = distance(layer.getNeuron(i).getOutput(), layer.getNeuron(j).getOutput());
                if(dist > maxDist){
                    maxDist = dist;
                }
            }
        }

        GaussianFunction.setSigma(maxDist, numNeurons);
    }

    @Override
    public double[] approximate(double[] inputs) {
        double[] outputs = new double[inputs.length];
        return outputs;

    }


    public void setMeans(List<Sample> sample){ //change to randomly select from samples
        Random random = new Random();
        for(int i = 0; i < numInputs; i++){
            means[i] = sample.get((int) random.nextDouble() * sample.size());
        }
    }

    public double distance(double x, double y){
        return Math.pow(Math.abs(x - y), 2);
    }

    public void updateWeights(){

    }
}

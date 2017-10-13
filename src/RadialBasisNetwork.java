import java.util.List;

public class RadialBasisNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private int numNeurons;
    private double[][] cluster;
    private double[] means; // store a mean for each cluster

    public RadialBasisNetwork(int inputs, int outputs, int numNeurons) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.numNeurons = numNeurons;
        cluster = new double[numNeurons][inputs];
        means = new double[numNeurons];
    }

    @Override
    public void train(List<Sample> samples) {
        // randomly split up data into clusters
        // calculate distances between each point in each cluster
        // reorganize data until all distances are minimized

    }

    @Override
    public double[] approximate(double[] inputs) {
        return new double[inputs.length];
        // for each cluster:
        // pass the value and the mean of current cluster to the gaussian()
        // keep track of which cluster has the smallest distance between value and mean
        // return the mean closest as the approximation
    }

    //get mean of cluster
    public double calcMean(int i) {
        double sum = 0.0;
        for (int j = 0; j < cluster[i].length; j++){
            sum += cluster[i][j];
        }
        return sum/cluster[i].length;
        //once training is done calculate the mean value for each cluster
        // and save to the means list
    }

    public void setMeans(){
        for(int i = 0; i < cluster.length; i++){
            means[i] = calcMean(i);
        }
    }

    public double distance(double x, double y){
        return Math.pow(Math.abs(x - y), 2);
    }
}

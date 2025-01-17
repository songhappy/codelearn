# BigDL related data structure
## BigDL scala related data structure

### BigDL Tensor, basic data structure to hold the data
```
trait Activity
class Table extends Activity
trait Tensor[T] extends Serializable with TensorMath[T] with Activity
class DenseTensor extends Tensor
```
BigDL Sample, MiniBatch
    Sample represents the features and labels of a data sample, features and labels are tensors.
    BigDL optimizer takes RDD[Sample[T]] or DataSet[D] currently. Eventually, everything should be packed into these two, I use RDD[Sample[T]] often.
    RDD[Sample[T]] is converted into Iterater[MiniBatch[T]] while building an Optimizer using SampleToMiniBatch transformer.
    //how to grab data from every node to build the MiniBatch?
BigDL DataSet and Transformer, all kinds of rdd manipulation
    DataSet are sent to Optimizer directly. It takes a transformer and manipulates the data. In the optimizer, it only caches the original dataset, and apply a couple of transformer later
```
Trait AbstractDataSet[D,DataSequence]{
  def transform[C: ClassTag](transformer: Transformer[D, C]): DataSet[C]
  def -> [C: ClassTag](transformer: Transformer[D, C]): DataSet[C] = {this.transform(transformer)}
}
```
ImageFrame, ImageFeature and Transformer
    It includes all image transformation, eventually to a list of ImageFeature which is a hashMap of sample and label(could be manipulated by BigDL optimizer and related) and other attributes could be manipulated by OpenCV
    Transformer defines a function which you want to apply to some data, Transformer can take an imageFrame and ImageFrame can take a transformer.
    DistributedImageFrame(var rdd: RDD[ImageFeature]) wraps RDD[ImageFeature]
    Eventually, features are array of numbers in dataframe to be sent to DLClassifier's fit
```
trait Transformer{
    def ->[C](other:Transformer): new ChainedTransformer(this, other) // take a transformer and chain it
}
abstract FeatureTransformer extends Transformer{
    def transform(feature:ImageFeature):ImageFeature //use OpenCV methodologies to transform a feature
    def apply(imageFrame: ImageFrame): ImageFrame = {
        imageFrame.transform(this)
     }
    override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
        prev.map(transform)
    }
}
```
```
Trait ImageFrame{
    def transform(transformer:Transformer):ImageFrame
    def ->(transformer:FeatureTransformer):ImageFrame = this.transform(transformer)
}

class DistributedImageFrame(var rdd: RDD[ImageFeature]) extends ImageFrame {
  override def transform(transformer: FeatureTransformer): ImageFrame = {
    rdd = transformer(rdd)  // = rdd.map(transformer.transform) = rdd.map(x=> transformer.transform(x))
    // rdd is an iterator?
    // transformer.apply function called twice?
    this
  }
}
```
```
class ImageFeature extends Serializable {
  private val state = new mutable.HashMap[String, Any]() // it uses HashMap to store all these data,original bytes read from image file, an opencv mat, pixels in float array, image label, sample, meta data and so on
  val sample = "sample"
  val label = "label"
}
```
```
val imageFrame: ImageFrame = ImageFrame.read(path, sqlContext.sparkContext)
    val transformer = Resize(256, 256) -> CenterCrop(224, 224) ->
      ChannelNormalize(123, 117, 104, 1, 1, 1) -> MatToTensor() -> ImageFrameToSample()
    val transformed1: ImageFrame = imageFrame.transform(transformer)
    val transformed2: ImageFrame = transformer(imageFrame) // transformer(imageFrame) = imageFrame.transform(transformer)
    // transformed1 and transformed2 are the same
```
### BigDL AbstractModule, all kinds of layers, Container for all models
```
abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable with InferShape{
abstract class TensorModule[T: ClassTag](implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T]
class Linear[T:ClassTag](val inputSize: Int, val outputSize: Int)(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable
```
```
abstract class Container[A <: Activity : ClassTag,B <: Activity : ClassTag, T: ClassTag](implicit ev: TensorNumeric[T]) extends AbstractModule[A, B, T]
abstract class DynamicContainer[A <: Activity : ClassTag, B <: Activity : ClassTag, T: ClassTag](implicit ev: TensorNumeric[T]) extends Container[A, B, T]
class Sequential[T: ClassTag](implicit ev: TensorNumeric[T]) extends DynamicContainer[Activity, Activity, T]
```

### BigDL dataset
### BigDL tensor
object TensorNumericMath {
    trait TensorNumeric[@specialized(Float, Double) T] extends Serializable 
    abstract class UndefinedTensorNumeric[@specialized(Float, Double) T](typeName: String) extends TensorNumeric
    
    object TensorNumeric {    
       implicit object NumericFloat extends UndefinedTensorNumeric[Float]("Float") {
          override def plus(x: Float, y: Float): Float = x + y
          ...}
        
    class DenseTensor(x: int, y:float)(implicit ev: TensorNumeric[T])) extends Tensor {
    override def methods from Tensor 
    }
   }
private[tensor] class DenseTensor[@specialized T: ClassTag](
  private[tensor] var _storage: ArrayStorage[T]...)(implicit ev: TensorNumeric[T]) extends Tensor[T] {}
    
object DenseTensor(
  def apply()
  and other utils functions. 
)


### BigDL training process
miniBatch: on each core
After each task computes its gradients, instead of sending gradients back to driver, gradients from all the partitions within a single worker are aggregated locally. Each node will have one gradient.
After that the aggregated gradient on each node is sliced into chunks and these chunks are exchanged between all the nodes in the cluster by block manager. 
Each node is responsible for a specific chunk, which in essence implements a PS architecture in BigDL for parameter synchronization. 
Each node retrieves gradients for the slice of the model that this node is responsible for from all the other nodes and aggregates them in multiple threads. 
After the pair-wise exchange completes, each node has its own portion of aggregated gradients and uses this to update its own portion of weights. 
Then the exchange happens again for synchronizing the updated weights. 

iteration, how to divide into different iterations?
how to for other nodes to get the updated parameters? each node reads updated weights from other nodes.


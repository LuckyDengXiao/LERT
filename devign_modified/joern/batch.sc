/*1:1:12,146,284*/

/*3:1:12,175*/

/*reveal:*/

/*cwe-362:123*/
/*cwe-362_:123*/
/*all:203,857,1145,1274,2051*/
import scala.util.control.Breaks._
@main def exec() = {
    var slice=0;
    for (slice <- 0 to 58){
        breakable{
            //if(slice==123){
             //   break()
            //}
            ///tf/devign/data_0_1times3
            loadCpg("/tf/devign/data_bigvul/cpg/"+slice+"_chatgpt_cpg.bin")
            cpg.runScript("/tf/devign/joern/graph-for-funcs.sc").toString() |> "/tf/devign/data_bigvul/cpg/"+slice+"_chatgpt_cpg.json"
            delete
        }
    }
}

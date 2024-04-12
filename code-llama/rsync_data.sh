#! /bin/bash

project_dir=$(cd "$(dirname $0)"; pwd)

function tf_hd() {
    echo rsync -avzulP --exclude "output_models*" $project_dir -e "ssh -p 8005" zzr@27.188.73.160:/data1/duanfuyao/tmp/
    rsync -avzulP --exclude "output_models*" $project_dir -e "ssh -p 8005" zzr@27.188.73.160:/data1/duanfuyao/tmp/

}

tf_hd 

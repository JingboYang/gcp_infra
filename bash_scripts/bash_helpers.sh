# Originally created for Stanford Spring 2019 CS341
# Jingbo Yang

SSH_KEY_LOC=$HOME/.ssh/YOUR_KEYLOCATION
GC_USERNAME=YOUR_USERNAME
GCHOME=/home/$GC_USERNAME
GC_HOME=$GCHMOE

echo "Available environment variables: GC_USERNAME, GCHOME, GC_HOME, SSH_KEY_LOC"

# Prints help statement
gc_help(){
    echo "gc_info"
    echo "\t\tReturn a list of GCP virtual machines"

    echo "gc_ssh VM_NAME"
    echo "\t\tSSH to a GCP instance"

    echo "gc_put VM_NAME LOCAL_DIR REMOTE_DIR"
    echo "\t\tUpload to remote directory"

    echo "gc_get VM_NAME REMOTE_DIR LOCAL_DIR"
    echo "\t\tDownload a remote directory"

    echo "gc_jupyter VM_NAME ROOT_DIR PORT"
    echo "\t\tLaunch jupyter notebok on remote"
}

# List all running virtual machines
gc_info() {
    gcloud compute instances list
}

# SSH in a VM instance
# Does the same thing as
#   gcloud compute ssh [INSTANCE_NAME]
# Usage:
#   gc_ssh pg-example
#   gc_ssh pg-example "ls /etc"
gc_ssh() {
    echo "Instance Name => $1"
    echo "Additional Args => $2"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    ssh -i $SSH_KEY_LOC $GC_USERNAME@$IP_ADDR $2
}

# Rsync upload to a specified VM
# Usage:
#   gc_put pg-example local_source_folder $GCHOME
gc_put() {
    echo "Instance Name => $1"
    echo "Source => $2"
    echo "Target => $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    rsync -Pavz -e "ssh -i $SSH_KEY_LOC" $2 $GC_USERNAME@$IP_ADDR:$3
}

# Rsync download from a specified VM
# Usage:
#   gc_get pg-example $GCHOME/source_code .
gc_get() {
    echo "Instance Name => $1"
    echo "Source => $2"
    echo "Target => $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")
    rsync -Pavz -e "ssh -i $SSH_KEY_LOC" $GC_USERNAME@$IP_ADDR:$2 $3
}

# Launch a Jupyter notebook at specified VM instance in specified folder using specified port
# Usage:
#   gc_jupyter pg-jupyter $GCHOME 8965
gc_jupyter() {
    echo "Instance Name => $1"
    echo "Root Directory => $2"
    echo "Port => $3"
    
    AFTER_SSH="source $GC_HOME/.bashrc; conda activate base; cd $2; jupyter notebook --no-browser --ip=\* --port $3"
    IP_ADDR=$(gcloud compute instances list --filter="name=$1" --format "get(networkInterfaces[0].accessConfigs[0].natIP)")

    echo "Jupyter notebook using will be available at "
    echo "============================================"
    echo -e "\t$IP_ADDR:$3"
    echo "============================================"

    gc_ssh $1 "-tt bash -l -c \"$AFTER_SSH\" "
}
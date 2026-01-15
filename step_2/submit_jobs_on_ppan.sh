#!/bin/bash

# ==========================
# Hyperparameter Configuration
# ==========================
# ENS_BLOCKS=(0 5 10 15)
ENS_BLOCKS=(10 15) # (5) (0 5)
# UMAP_MDS=(0.1 0.3 0.5 0.7 0.9)
UMAP_MDS=(0.7 0.9) #(0.1 0.3 0.5) #
# UMAP_NNS=(5 10 50 100 200)
UMAP_NNS=(5 10 50) #(100 200) #
# NUM_CLUSTS=(3 5 6 7 10 13 15 16 20 25)
NUM_CLUSTS=(6 15)
HCLUST_N=120


###############################################################################
# Paths
###############################################################################
DATA_RES="p25"
DATA_FIELD="dynamics"

BASE_DIR="/work/lnd/CM4X/NEMI/CM4X-${DATA_RES}/outputs/${DATA_FIELD}"
INDIR="${BASE_DIR}/embeddings"

# ==========================
# Submit Jobs
# ==========================
for UMAP_MD in "${UMAP_MDS[@]}"; do
    for UMAP_NN in "${UMAP_NNS[@]}"; do
        for MEMBER in "${ENS_BLOCKS[@]}"; do
			for NC in "${NUM_CLUSTS[@]}"; do
				OUTDIR="${BASE_DIR}/clusterings/nclusters_${NC}"
				mkdir -p "${OUTDIR}"
				# Loop through the ensemble numbers from 1 to 5
				echo
				for i in {1..5}; do
					echo "================================"
					# Increment the ensemble number
					MEMBER=$((MEMBER + 1))
					echo "Submitting: ENS_MEMBER=${MEMBER}, md=${UMAP_MD}, nn=${UMAP_NN}, nclusters=${NC}"
					sbatch \
						--job-name=HAC_MEMBER:${MEMBER}_MD:${UMAP_MD}_NN:${UMAP_NN}_NC:${NC} \
						clustering_on_ppan.sh \
						"${INDIR}" \
						"${UMAP_MD}" \
						"${UMAP_NN}" \
						"${MEMBER}" \
						"${HCLUST_N}" \
						"${NC}" \
						"${OUTDIR}"
					echo "================================"
				done
				echo
			done
        done
    done
done
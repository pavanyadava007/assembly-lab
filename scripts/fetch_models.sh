#!/bin/bash
# Fetch the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0) at the commit used for all results.
set -e
cd "$(dirname "$0")/../models"
[ -d menagerie ] || git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
cd menagerie && git sparse-checkout set franka_emika_panda && git checkout -q 4d038b3feae26ec82b46a4d586379114012a8ac7
echo "Panda model ready at models/menagerie/franka_emika_panda"

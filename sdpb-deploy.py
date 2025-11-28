#!/usr/bin/env python

import argparse
import copy
from datetime import datetime
import git
import os
import shutil
import subprocess as sp
import tempfile
from enum import StrEnum, auto

HOME = os.environ['HOME']


class Cluster(StrEnum):
    BOSTON = auto()
    EXPANSE = auto()
    IMPERIAL = auto()
    UNKNOWN = auto()


def detect_cluster() -> Cluster:
    if os.path.exists('/expanse/lustre/'):
        return Cluster.EXPANSE
    elif HOME.startswith('/usr2/collab'):
        return Cluster.BOSTON
    elif HOME.startswith('/rds/general/user/'):
        return Cluster.IMPERIAL
    else:
        return Cluster.UNKNOWN


def get_num_threads(cluster: Cluster) -> int:
    num_threads = 1
    if cluster == Cluster.BOSTON:
        num_threads = int(os.environ.get('NSLOTS', 1))
    elif cluster == Cluster.EXPANSE or cluster == cluster.IMPERIAL:
        if os.environ['HOSTNAME'].startswith('login'):
            num_threads = 1
        else:
            num_threads = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', 16))
    else:
        raise ValueError(f'Unknown cluster: {cluster}')

    if num_threads > 1:
        num_threads -= 1
    return num_threads


# TODO: Store as Namespace? Read from file?
CONFIG = {Cluster.BOSTON: {
    'modules': {'default': [
        'python3', 'gcc/12.2.0', 'openmpi/4.1.5', 'cmake/3.22.2', 'gmplib/6.2.1', 'mpfr/4.2.1',
        'boost/1.83.0',
        # 'boost/1.79.0_gcc5+',
    ], 'flint': ['openblas/0.3.23'],
        'mpsolve': []
    },
    'waf_args': {'default': [
        '--elemental-dir=$HOME/install',
        '--libarchive-dir=$HOME/install',
        '--gmpxx-dir=$SCC_GMPLIB_DIR',
        '--mpfr-dir=$SCC_MPFR_DIR',
        '--rapidjson-dir=$HOME/install',
    ],
        'flint': ['--flint-dir=$HOME/install', '--cblas-dir=$SCC_OPENBLAS_DIR'],
        'mpsolve': ['--mpsolve-dir=$HOME/install']},
    # Set environment variables for ./waf configure
    'waf_env': 'OMPI_LDFLAGS="$(mpicxx --showme:link) -Wl,--disable-new-dtags"',
    # lua or tcl
    'module_lang': 'lua',
    'mpirun': 'mpirun',
    # sbatch or qsub
    'sbatch': 'qsub',
    '/bin/sh': '/bin/sh -l',
    'SLURM_SUBMIT_DIR': 'SGE_O_WORKDIR',
    'sbatch_config_template': '''
# SGE configuration, see https://www.bu.edu/tech/support/research/system-usage/running-jobs
# 1 node, {num_cpus} CPUs
# 30 min time limit
# merge stderr into stdout
#
#$ -pe mpi_{num_cpus}_tasks_per_node {num_cpus}
#$ -l h_rt=00:30:00
#$ -j y
'''
},
    Cluster.EXPANSE: {
    'modules': {'default': [
        'cpu/0.15.4',
        'gcc/10.2.0',
        'openmpi/4.0.4',
        'gmp/6.1.2',
        'mpfr/4.0.2',
        'slurm',
    ], 'flint': ['openblas/dynamic/0.3.7'],
        'mpsolve': []
    },
    'waf_args': {'default': [
        '--rapidjson-dir=$HOME/install',
        '--elemental-dir=$HOME/install',
        '--rapidjson-dir=$HOME/install',
        # '--boost-dir=$HOME/install',
        '--libarchive-dir=$HOME/install'
    ], 'flint': ['--flint-dir=$HOME/install'], 'mpsolve': ['--mpsolve-dir=$HOME/install']},
    # Set environment variables for ./waf configure
    'waf_env': 'CXX=mpicxx',
    # lua or tcl
    'module_lang': 'lua',
    'mpirun': 'srun -v',
    # sbatch or qsub
    'sbatch': 'sbatch',
    'SLURM_SUBMIT_DIR': 'SLURM_SUBMIT_DIR',
    'sbatch_config_template': '''
# SLURM configuration, see https://www.sdsc.edu/support/user_guides/expanse.html#running and https://slurm.schedmd.com/sbatch.html
# 1 node, {num_cpus} CPUs, 16GB RAM
# 30 min time limit
#
#SBATCH --job-name="sdpb_example.sh"
#SBATCH --output="sdpb_example.sh.%j.out"
#SBATCH --no-requeue
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_cpus}
#SBATCH --mem=16G
#SBATCH --account=yun124
#SBATCH -t 00:30:00'''
},
    Cluster.IMPERIAL: {
    'modules': {'default': [
        'tools/prod',
        # 'Autotools/20220317-GCCcore-12.3.0',
        'Boost/1.82.0-GCC-12.3.0',
        # 'CMake/3.26.3-GCCcore-12.3.0',
        'GCC/12.3.0',
        'GMP/6.2.1-GCCcore-12.3.0',
        'libarchive/3.6.2-GCCcore-12.3.0',
        'METIS/5.1.0-GCCcore-12.3.0',
        'MPFR/4.2.0-GCCcore-12.3.0',
        # NB: Elemental is built with openblas, so we need it even before 3.0.0
        'OpenBLAS/0.3.23-GCC-12.3.0',
        'OpenMPI/4.1.5-GCC-12.3.0',
        'pkg-config/0.29.2-GCCcore-12.3.0',
        'Python/3.11.3-GCCcore-12.3.0',
        'RapidJSON/1.1.0-20230928-GCCcore-12.3.0'
    ],
        'flint': [], 'mpsolve': []
    },
    'waf_args': {'default': ['--elemental-dir=$HOME/install/cx3'],
                 'flint': ['--flint-dir=$HOME/install/cx3'],
                 'mpsolve': ['--mpsolve-dir=$HOME/install/cx3']},
    # Set environment variables for ./waf configure
    'waf_env': 'CXX=mpicxx',
    # lua or tcl
    'module_lang': 'lua',
    'mpirun': 'mpirun',
    # sbatch or qsub
    'sbatch': 'qsub',
    'SLURM_SUBMIT_DIR': 'PBS_O_WORKDIR',
    'sbatch_config_template': '''
# PBSPro configuration, see https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/mpi-jobs/
# 1 node, {num_cpus} CPUs, 16GB RAM
# 30 min time limit
# merge stderr into stdout
#
#PBS -l select=1:ncpus={num_cpus}:mpiprocs={num_cpus}:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
'''
},
}


def run(cmd) -> sp.CompletedProcess[bytes]:
    # Multiple commands
    if isinstance(cmd, list):
        cmd_lines = '\n'.join(cmd)
        print(
            f'===Preparing to run shell commands (joined by &&): \n{cmd_lines}')
        print('===')
        cmd = " && ".join(cmd)
    print(cmd)
    return sp.run(cmd, shell=True, check=True)


def git_clone_sdpb(args):
    """Checkout the specified branch or tag."""
    target = args.version
    # Check if the target is a remote branch (e.g., upstream/master)
    if "/" in target:
        remote, branch = target.split("/", 1)
    else:
        # remote = "upstream"
        remote = "davidsd"
        branch = target

    # 3.0.0 does not compile on Imperial with FLINT, so we checkout a later commit.
    # TODO: should we make a 3.0.1 release with such small fixes?
    if args.cluster == Cluster.IMPERIAL and branch == '3.0.0':
        branch = 'c5736a6e'
    run([
        f'cd {args.tmp_dir}',
        f'git clone https://github.com/{remote}/sdpb.git',
        'cd sdpb',
        'git fetch --all --tags',
        f'git checkout {branch}',
    ])


def parse_args() -> list[argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument('versions', type=str,
                        help='Version or comma-separated list of versions to install, i.e. branch or tag. Examples: 3.0.0, upstream/master, vasdommes/simbpleboot-pmp-json.')
    # parser.add_argument('--source-dir', default=f'{HOME}/sdpb',
    #                     help=f'Source directory, e.g. {HOME}/sdpb')
    parser.add_argument('--install-prefix', default=f'{HOME}/install',
                        help='Root directory for installation. (default: %(default)s)')
    parser.add_argument('--install-name', required=False,
                        help='Installation name, e.g. sdpb/3.0.0 (default: sdpb/version). Full path to the SDPB binary is INSTALL_PREFIX/INSTALL_NAME/bin/sdpb')
    parser.add_argument('--module-prefix', default=f'{HOME}/install/modules',
                        help='Modules prefix (default: %(default)s)')
    parser.add_argument('--module-name', required=False,
                        help='Module name, e.g. sdpb/3.0.0 (default: --module-name = --install-name). Installs module MODULES_PREFIX/MODULE_NAME')
    parser.add_argument('-v', '--verbose', help='increase output verbosity',
                        action='store_true')
    original_args = parser.parse_args()
    print(f'original_args={original_args}')

    cluster = detect_cluster()
    print(f'Detected cluster={cluster}')
    original_args.cluster = cluster
    if cluster == Cluster.UNKNOWN:
        raise ValueError('Unknown cluster!')

    def create_args_for_version(version):
        args = copy.deepcopy(original_args)
        del args.versions  # we don't need it, we'll have a single version instead
        args.version = version
        if not args.install_name:
            args.install_name = f'sdpb/{args.version}'

        if not args.install_name.startswith('sdpb/'):
            # TODO: print warning instead?
            raise ValueError(
                f'--install-name={args.install_name} should start with sdpb/')
        if not args.module_name:
            args.module_name = args.install_name
        return args

    versions = [v.strip() for v in original_args.versions.strip().split(',')]
    args_list = [create_args_for_version(v) for v in versions]
    print(f'args_list={args_list}')
    return args_list


def waf(args: argparse.Namespace):
    def has_commit(commit_hash):
        return args.repo.is_ancestor(commit_hash, 'HEAD')

    print('===Load modules and run waf')
    install_dir = os.path.join(args.install_prefix, args.install_name)

    config = CONFIG[args.cluster]

    modules = config['modules']['default']

    waf_configure_args = [f'--prefix={install_dir}'] + \
        config['waf_args']['default']

    has_flint = has_commit('4368aeb8')
    if has_flint:
        modules += config['modules']['flint']
        waf_configure_args += config['waf_args']['flint']

    # El::Matrix::Control() causes memory leaks
    has_El_Matrix_Control = has_commit(
        '407169c1') and not has_commit('09a16ddb')
    if has_El_Matrix_Control:
        raise ValueError(
            'This SDPB version requires El::Matrix::Control(), which has been removed from the current Elemental installation.')

    has_mpsolve = has_commit('6eb5df35') or has_commit(
        '4d0b56f7') or 'mpsolve' in args.version
    if has_mpsolve:
        modules += config['modules']['mpsolve']
        waf_configure_args += config['waf_args']['mpsolve']

    num_threads = get_num_threads(args.cluster)
    print(f'num_threads={num_threads}')

    # Before updating from 2.0.14 to 2.0.26, ./waf failed
    call_waf = './waf' if has_commit('ae3801ae') else 'python2 ./waf'

    if os.path.exists(install_dir):
        parent, name = os.path.split(install_dir)
        backup_dir = os.path.join(
            parent, '.backup', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), name)
        print(f'Backup: copy {install_dir} to {backup_dir}')
        shutil.copytree(install_dir, backup_dir)

    # NB: we have to run all commands in a single shell, otherwise "module load" command will not take effect.
    run([
        f'cd {args.source_dir}',
        f'module purge',
        f'module load {" ".join(modules)}',
        f'{call_waf} --version',
        # 'rm -rf ./build',
        f'({config["waf_env"]} {call_waf} configure {" ".join(waf_configure_args)} || (cat build/config.log && exit 1))',
        f'{call_waf} -j{num_threads}',
        'ldd ./build/sdpb',
        './build/sdpb --version',
        f'{call_waf} install'
    ])


def write_module(args: argparse.Namespace):
    lang = CONFIG[args.cluster]['module_lang']

    if lang == 'lua':
        module_filename = f'{args.module_name}.lua'
        common_filename = 'common.lua'
    elif lang == 'tcl':
        module_filename = args.module_name
        common_filename = 'common.tcl'
    else:
        raise ValueError(f'Unknown module language: {lang}')

    module_path = os.path.join(args.module_prefix, module_filename)
    print(f'===Writing modulefile: {module_path}')
    sdpb = args.module_name.split('/')[0]
    if sdpb != 'sdpb':
        raise ValueError(
            f'Module name {args.module_name} does not start with "sdpb/"')

    common_path = os.path.join(
        args.module_prefix, sdpb, f'.common/{common_filename}')
    print(f'{common_filename} path: {common_path}')

    if not os.path.exists(common_path):
        raise ValueError(
            f'Failed to create modulefile, {common_filename} path does not exist: {common_path}')

    os.makedirs(os.path.dirname(module_path), exist_ok=True)

    if lang == 'lua':
        module_text = f'''-- Load common.lua directly using loadfile
local common, err = loadfile("{common_path}")
if not common then
    LmodError("Failed to load common.lua: " .. err)
end
common = common()  -- Execute the loaded file

common.setup_sdpb()
'''
    elif lang == 'tcl':
        module_text = f'''#%Module
source {common_path}
setup_sdpb
'''
    else:
        raise ValueError(f'Unknown module language: {lang}')

    with open(module_path, 'w') as f:
        f.write(module_text)

# pmp2sdp has been added in 2.7.0


def write_example(args: argparse.Namespace):
    def has_commit(commit_hash):
        return args.repo.is_ancestor(commit_hash, 'HEAD')

    config = CONFIG[args.cluster]
    script_path = os.path.join(
        args.install_prefix, args.install_name, 'share', 'sdpb_example.sh')
    data_dir = os.path.join(
        args.install_prefix, 'share/sdpb/SingletScalar_cT_test_nmax6/primal_dual_optimal')
    if not os.path.exists(data_dir):
        raise ValueError(f'Directory does not exist: {data_dir}')

    # NB: num_cpus=6 won't work for BOSTON
    num_cpus = 16
    procs_per_node = '' if has_commit(
        '5aa622c8') else f'--procsPerNode={num_cpus}'

    for pmp2sdp_name in ['pmp2sdp', 'sdp2input']:
        pmp2sdp_path = os.path.join(
            args.install_prefix, args.install_name, 'bin', pmp2sdp_name)
        if os.path.exists(pmp2sdp_path):
            break

    if not os.path.exists(pmp2sdp_path):
        raise ValueError(f'File not found: {pmp2sdp_path}')

    # When reading .nsv files, treat relative paths as relative to .nsv directory.
    # Old version (2.5.1) didn't do it,
    optional_cd_to_input_json_dir = '' if has_commit(
        '156fc7ec') else 'cd $DATA_DIR/json'

    mpirun = config['mpirun']
    bin_sh = config.get('/bin/sh', '/bin/sh')
    sbatch = config['sbatch']
    SLURM_SUBMIT_DIR = config['SLURM_SUBMIT_DIR']

    sbatch_config = config["sbatch_config_template"].format(num_cpus=num_cpus)

    # Usage: sbatch run_sdpb_example.sh
    script_text = f'''#!{bin_sh}

# Example script which runs {pmp2sdp_name} and sdpb
# Data taken from:
# https://github.com/davidsd/sdpb/tree/8f127d92078b04e36ab587e7e4344410c26a2a5f/test/data/end-to-end_tests/SingletScalar_cT_test_nmax6/primal_dual_optimal
# Usage: {sbatch} run_sdpb_example.sh

{sbatch_config}

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

if [ -z "${{{SLURM_SUBMIT_DIR}+x}}" ]; then
    echo "{SLURM_SUBMIT_DIR} not set. Run this script as: {sbatch} sdpb_example.sh"
    exit 1
fi

echo $(date +"[%a %Y-%m-%d %H:%M:%S]")


module purge
module use {args.module_prefix}
module load {args.module_name}

DATA_DIR={data_dir}
OUTPUT_DIR=${SLURM_SUBMIT_DIR}

echo SDPB_VERSION=$SDPB_VERSION
echo SDPB_HOME=$SDPB_HOME
echo DATA_DIR=$DATA_DIR
echo OUTPUT_DIR=$OUTPUT_DIR
echo "==========================="

{optional_cd_to_input_json_dir}
RUN_PMP2SDP="{mpirun} {pmp2sdp_name} --input=$DATA_DIR/json/file_list.nsv --output=$OUTPUT_DIR/sdp --precision=768"

echo $RUN_PMP2SDP
$RUN_PMP2SDP

RUN_SDPB="{mpirun} sdpb {procs_per_node} --checkpointInterval 3600 --maxRuntime 1800 --dualityGapThreshold 1.0e-30 --primalErrorThreshold 1.0e-30 --dualErrorThreshold 1.0e-30 --initialMatrixScalePrimal 1.0e20 --initialMatrixScaleDual 1.0e20 --feasibleCenteringParameter 0.1 --infeasibleCenteringParameter 0.3 --stepLengthReduction 0.7 --maxComplementarity 1.0e100 --maxIterations 1000 --verbosity 1 --procGranularity 1 --writeSolution x,y --checkpointDir=$OUTPUT_DIR/ck --outDir=$OUTPUT_DIR/out --precision=768 --sdpDir=$OUTPUT_DIR/sdp"

echo $RUN_SDPB
$RUN_SDPB

echo "==========================="
echo $(date +"[%a %Y-%m-%d %H:%M:%S]")
echo "sdpb_example.sh finished. Please compare your solution with $DATA_DIR/out"
'''

    print(f'Writing example script to {script_path}')
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_text)


def run_example(args: argparse.Namespace):
    sbatch = CONFIG[args.cluster]['sbatch']
    script_path = os.path.join(
        args.install_prefix, args.install_name, 'share', 'sdpb_example.sh')
    job_dir = os.path.join(f'{HOME}/deploy/{sbatch}', args.install_name,
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(job_dir)
    print(f'===Run example script {script_path} at {job_dir}')
    run([f'cd {job_dir}', f'{sbatch} {script_path}'])


def main():
    args_list = parse_args()
    # versions = [x.strip() for x in args.versions.strip('"').split(',')]
    print(f'versions={[args.version for args in args_list]}')
    for args in args_list:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f'===Deploy version={args.version}')
            args.tmp_dir = tmp_dir
            args.source_dir = os.path.join(tmp_dir, 'sdpb')
            print(f'args={args}')
            git_clone_sdpb(args)
            args.repo = git.Repo(args.source_dir)

            # # TODO for debug:
            run([
                f'cd {args.source_dir}',
                'git describe --tags --always --dirty',
                'git rev-parse --abbrev-ref HEAD',
                'git log | head -n 1',
            ])
            # continue

            waf(args)
            write_module(args)
            write_example(args)
            run_example(args)

            # Print module info
            run([
                f'module use {args.module_prefix}',
                f'module load {args.module_name}',
                'echo $SDPB_VERSION',
                'sdpb --version | head -n 1'
            ])

            print(f'===Deploy finished for version={args.version}')

    print('Deploy finished')


if __name__ == '__main__':
    main()

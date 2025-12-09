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
    CALTECH = auto()
    EXPANSE = auto()
    IMPERIAL = auto()
    YALE = auto()
    UNKNOWN = auto()


def detect_cluster() -> Cluster:
    if os.path.exists('/resnick/'):
        return Cluster.CALTECH
    if os.path.exists('/expanse/lustre/'):
        return Cluster.EXPANSE
    if os.path.exists('/vast/palmer/home.grace/'):
        return Cluster.YALE
    if HOME.startswith('/usr2/collab'):
        return Cluster.BOSTON
    if HOME.startswith('/rds/general/user/'):
        return Cluster.IMPERIAL

    return Cluster.UNKNOWN


def get_num_threads(cluster: Cluster) -> int:
    num_threads = 1
    if cluster == Cluster.BOSTON:
        num_threads = int(os.environ.get('NSLOTS', 1))
    elif cluster in [Cluster.CALTECH, Cluster.EXPANSE, Cluster.IMPERIAL, Cluster.YALE]:
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
CONFIG = {
    Cluster.IMPERIAL: {
        'modules': [
            'tools/prod',
            'Eigen/3.4.0-GCCcore-11.3.0',
            # NB: Trilinos depends on GCCcore/11.3.0,
            # so we cannot load e.g. Eigen/3.4.0-GCCcore-14.3.0
            'Trilinos/13.4.1-foss-2022a',
        ],
        'waf_args': ['--eigen-dir=$EBROOTEIGEN',
                     '--trilinos-dir=$EBROOTTRILINOS',
                     '--trilinos-libs="teuchoscomm teuchoscore teuchosparameterlist kokkoscore"',
                     ],
        # Set environment variables for ./waf configure
        'waf_env': 'CXX=mpicxx CXXFLAGS="-fopenmp" LDFLAGS="-fopenmp -ldl"',
        # lua or tcl
        'module_lang': 'lua',
        # sbatch or qsub
        'sbatch': 'qsub',
        'SLURM_SUBMIT_DIR': 'PBS_O_WORKDIR',
        'sbatch_config_template': '''
# PBSPro configuration, see https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/running-your-first-job/
# 1 node, {num_cpus} CPUs, 16GB RAM
# 30 min time limit
# merge stderr into stdout
#
#PBS -l select=1:ncpus={num_cpus}:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
'''
    },
    Cluster.YALE: {
        'modules': [
            'Eigen/3.4.0-GCCcore-12.2.0',
            'Trilinos/13.4.1-foss-2022b-zoltan',
        ],
        'waf_args': [
            # '--eigen-dir=$EBROOTEIGEN',
            #  '--trilinos-dir=$EBROOTTRILINOS',
            '--trilinos-libs="teuchoscomm teuchoscore teuchosparameterlist kokkoscore"',
        ],
        # Set environment variables for ./waf configure
        'waf_env': 'CXX=mpicxx CXXFLAGS="-fopenmp" LDFLAGS="-fopenmp -ldl"',
        # lua or tcl
        'module_lang': 'lua',
        # sbatch or qsub
        'sbatch': 'sbatch',
        'SLURM_SUBMIT_DIR': 'SLURM_SUBMIT_DIR',
        'sbatch_config_template': '''
# SLURM configuration, see https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/ and https://slurm.schedmd.com/sbatch.html
# 1 node, {num_cpus} CPUs, 16GB RAM
# 30 min time limit
#
#SBATCH --job-name="scalar_blocks_example.sh"
#SBATCH --output="scalar_blocks_example.sh.%j.out"
#SBATCH --no-requeue
#SBATCH --partition=pi_poland
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_cpus}
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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


def git_clone_scalar_blocks(args):
    """Checkout the specified branch or tag."""
    target = args.version
    # Check if the target is a remote branch (e.g., upstream/master)
    if "/" in target:
        remote, branch = target.split("/", 1)
    else:
        # remote = "upstream"
        remote = "bootstrapcollaboration"
        branch = target

    run([
        f'cd {args.tmp_dir}',
        f'git clone https://gitlab.com/{remote}/scalar_blocks.git',
        'cd scalar_blocks',
        'git fetch --all --tags',
        f'git checkout {branch}',
    ])


def parse_args() -> list[argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument('versions', type=str,
                        help='Version or comma-separated list of versions to install, i.e. branch or tag. Examples: 1.0.1, upstream/master, vasdommes/master.')
    parser.add_argument('--install-prefix', default=f'{HOME}/install',
                        help='Root directory for installation. (default: %(default)s)')
    parser.add_argument('--install-name', required=False,
                        help='Installation name, e.g. scalar_blocks/1.0.1 (default: scalar_blocks/version). Full path to the binary is INSTALL_PREFIX/INSTALL_NAME/bin/scalar_blocks')
    parser.add_argument('--module-prefix', default=f'{HOME}/install/modules',
                        help='Modules prefix (default: %(default)s)')
    parser.add_argument('--module-name', required=False,
                        help='Module name, e.g. scalar_blocks/1.0.1 (default: --module-name = --install-name). Installs module MODULES_PREFIX/MODULE_NAME')
    parser.add_argument('-v', '--verbose', help='increase output verbosity',
                        action='store_true')
    original_args = parser.parse_args()
    print(f'original_args={original_args}')

    cluster = detect_cluster()
    print(f'Detected cluster={cluster}')
    original_args.cluster = cluster
    if cluster == Cluster.UNKNOWN:
        raise ValueError('Unknown cluster!')
    if cluster not in CONFIG:
        raise ValueError(f'No configuration found for cluster={cluster}')

    def create_args_for_version(version):
        args = copy.deepcopy(original_args)
        del args.versions  # we don't need it, we'll have a single version instead
        args.version = version
        if not args.install_name:
            args.install_name = f'scalar_blocks/{args.version}'

        if not args.install_name.startswith('scalar_blocks/'):
            # TODO: print warning instead?
            raise ValueError(
                f'--install-name={args.install_name} should start with scalar_blocks/')
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

    modules = config['modules']

    waf_configure_args = [f'--prefix={install_dir}'] + config['waf_args']

    num_threads = get_num_threads(args.cluster)
    print(f'num_threads={num_threads}')

    # Before updating from 2.0.14 to 2.0.26, ./waf failed
    call_waf = './waf' if has_commit('e0610f17') else 'python2 ./waf'

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
        'ldd ./build/scalar_blocks',
        './build/scalar_blocks --version',
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
    scalar_blocks = args.module_name.split('/')[0]
    if scalar_blocks != 'scalar_blocks':
        raise ValueError(
            f'Module name {args.module_name} does not start with "scalar_blocks/"')

    common_path = os.path.join(
        args.module_prefix, scalar_blocks, f'.common/{common_filename}')
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

common.setup_scalar_blocks()
'''
    elif lang == 'tcl':
        module_text = f'''#%Module
source {common_path}
setup_scalar_blocks
'''
    else:
        raise ValueError(f'Unknown module language: {lang}')

    with open(module_path, 'w') as f:
        f.write(module_text)


def write_example(args: argparse.Namespace):
    config = CONFIG[args.cluster]
    script_path = os.path.join(
        args.install_prefix, args.install_name, 'share', 'scalar_blocks_example.sh')
    share_example_out_dir = os.path.join(
        args.install_prefix, 'share/scalar_blocks/example_out')
    if not os.path.exists(share_example_out_dir):
        raise ValueError(f'Directory does not exist: {share_example_out_dir}')

    # NB: num_cpus=6 won't work for BOSTON
    num_cpus = 16

    bin_sh = config.get('/bin/sh', '/bin/sh')
    sbatch = config['sbatch']
    SLURM_SUBMIT_DIR = config['SLURM_SUBMIT_DIR']

    sbatch_config = config["sbatch_config_template"].format(num_cpus=num_cpus)

    # Usage: sbatch scalar_blocks_example.sh
    script_text = f'''#!{bin_sh}

# Example script which runs scalar_blocks
# Based on:
# https://gitlab.com/bootstrapcollaboration/scalar_blocks/blob/91678587bbbb562902b77a617a36df69f07cb494/tests/create_test_tables.sh
# Usage: {sbatch} scalar_blocks_example.sh

{sbatch_config}

# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

if [ -z "${{{SLURM_SUBMIT_DIR}+x}}" ]; then
    echo "{SLURM_SUBMIT_DIR} not set. Run this script as: {sbatch} scalar_blocks_example.sh"
    exit 1
fi

echo $(date +"[%a %Y-%m-%d %H:%M:%S]")


module purge
module use {args.module_prefix}
module load {args.module_name}

OUTPUT_DIR=${SLURM_SUBMIT_DIR}

echo SCALAR_BLOCKS_VERSION=$SCALAR_BLOCKS_VERSION
echo SCALAR_BLOCKS_HOME=$SCALAR_BLOCKS_HOME
echo OUTPUT_DIR=$OUTPUT_DIR
echo "==========================="

RUN_SCALAR_BLOCKS="scalar_blocks --dim 6 --order 60 --max-derivs 6 --spin-ranges 0-10 --poles 20 --delta-12 0.8791  --delta-34 1.5791 --num-threads=4 -o $OUTPUT_DIR --precision=665"

echo $RUN_SCALAR_BLOCKS
$RUN_SCALAR_BLOCKS

echo "==========================="
echo $(date +"[%a %Y-%m-%d %H:%M:%S]")
echo "scalar_blocks_example.sh finished. Please compare your solution with {share_example_out_dir}"
'''

    print(f'Writing example script to {script_path}')
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_text)


def run_example(args: argparse.Namespace):
    sbatch = CONFIG[args.cluster]['sbatch']
    script_path = os.path.join(
        args.install_prefix, args.install_name, 'share', 'scalar_blocks_example.sh')
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
            args.source_dir = os.path.join(tmp_dir, 'scalar_blocks')
            print(f'args={args}')
            git_clone_scalar_blocks(args)
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
                'echo $SCALAR_BLOCKS_VERSION',
                'scalar_blocks --version'
            ])

            print(f'===Deploy finished for version={args.version}')

    print('Deploy finished')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
# SPDX-License-Identifier: Apache-2.0

"""Resolve project releases and publish verified TVM image indexes."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def output(values):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        for key, value in values.items():
            stream.write(f'{key}={value}\n')


def resolve(args, config):
    release = args.release
    if not re.fullmatch(r'\d+\.\d+\.\d+', release):
        raise ValueError('Project release must be X.Y.Z, for example 0.16.1')
    result = subprocess.run(['git', 'rev-parse', '--verify', f'refs/tags/{release}^{{commit}}'],
                            capture_output=True, text=True)
    if result.returncode:
        if args.publish:
            raise ValueError('Publication requires an existing project release tag')
        source = command('git', 'rev-parse', 'HEAD')
    else:
        source = result.stdout.strip()
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Invalid source commit')
    if args.publish:
        subprocess.run(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], check=True)
    if getattr(args, 'config', None):
        config_path = Path(args.config).resolve().relative_to(Path.cwd().resolve()).as_posix()
        tagged_config = subprocess.run(['git', 'show', f'{source}:{config_path}'],
                                       capture_output=True, text=True)
        if tagged_config.returncode == 0:
            config = json.loads(tagged_config.stdout)
        elif args.publish and not (
            release == config.get('legacy_release_tag')
            and source == config.get('legacy_source_sha')
        ):
            raise ValueError('Release tag does not contain its dependency configuration')
        validate_config(config)
    cpu_tag = f'{release}-tvm{config["tvm_version"]}'
    output({'release': release, 'source_sha': source, 'cpu_tag': cpu_tag,
            'cuda_tag': f'{cpu_tag}-cuda{config["cuda_version"]}',
            **{key: config[key] for key in ['tvm_version', 'ffi_version', 'tvm_commit', 'ffi_commit', 'cuda_version']}})


def inspect_optional(reference):
    result = subprocess.run(['docker', 'manifest', 'inspect', reference],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    if any(error in result.stderr.lower() for error in ('manifest unknown', 'no such manifest')):
        return None
    raise ValueError(f'Cannot inspect {reference}: {result.stderr.strip()}')


def platforms(index):
    result = {}
    for descriptor in index.get('manifests', []):
        platform = descriptor.get('platform', {})
        arch = platform.get('architecture')
        if platform.get('os') != 'linux':
            raise ValueError('Unexpected image platform')
        if arch == 'arm':
            if platform.get('variant') != 'v7':
                raise ValueError('Expected ARMv7')
            arch = 'armv7'
        if arch in result:
            raise ValueError('Duplicate image platform')
        result[arch] = descriptor['digest']
    return result


def manifest(args, config):
    expected_tag = f'{args.release}-tvm{config["tvm_version"]}'
    if args.backend == 'cuda':
        expected_tag += f'-cuda{config["cuda_version"]}'
    if args.tag != expected_tag or not re.fullmatch(r'\d+\.\d+\.\d+', args.release):
        raise ValueError('Image tag does not match the project release and backend')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_sha):
        raise ValueError('Invalid source commit')
    architectures = args.architectures.split(',')
    directory = Path(args.digests)
    if {p.name for p in directory.iterdir()} != {f'{arch}.txt' for arch in architectures}:
        raise ValueError('Unexpected digest artifacts')
    references = []
    expected = {}
    for arch in architectures:
        reference = (directory / f'{arch}.txt').read_text().strip()
        if not re.fullmatch(re.escape(args.image) + r'@sha256:[0-9a-f]{64}', reference):
            raise ValueError(f'Invalid image digest for {arch}')
        image = json.loads(command('docker', 'buildx', 'imagetools', 'inspect',
                                   '--format', '{{json .Image}}', reference))
        platform_arch = 'arm' if arch == 'armv7' else arch
        if (image.get('os'), image.get('architecture')) != ('linux', platform_arch):
            raise ValueError(f'Incorrect platform for {arch}')
        if arch == 'armv7' and image.get('variant') != 'v7':
            raise ValueError('Incorrect ARM variant')
        labels = image.get('config', {}).get('Labels', {})
        required = {
            'org.opencontainers.image.source': 'https://github.com/' + os.environ['GITHUB_REPOSITORY'],
            'org.digitalhub.release.version': args.release,
            'org.digitalhub.source.revision': args.source_sha,
            'org.opencontainers.image.version': config['tvm_version'],
            'org.opencontainers.image.revision': config['tvm_commit'],
            'org.digitalhub.tvm-ffi.version': config['ffi_version'],
            'org.digitalhub.tvm-ffi.revision': config['ffi_commit'],
        }
        if args.backend == 'cuda':
            required['org.digitalhub.cuda.version'] = config['cuda_version']
        elif labels.get('org.digitalhub.cuda.version'):
            raise ValueError('CPU image unexpectedly declares CUDA')
        for key, value in required.items():
            if labels.get(key) != value:
                raise ValueError(f'{arch}: incorrect {key}: {labels.get(key)!r}')
        references.append(reference)
        expected[arch] = reference.split('@', 1)[1]
    target = f'{args.image}:{args.tag}'
    existing = inspect_optional(target)
    if existing is not None:
        if platforms(existing) != expected:
            raise ValueError(f'{target} already exists with different digests; refusing overwrite')
        print('Existing manifest already contains the verified digests')
    else:
        preview = command('docker', 'buildx', 'imagetools', 'create', '--dry-run', *references)
        if platforms(json.loads(preview)) != expected:
            raise ValueError('Manifest preview does not match the image digests')
        if not args.publish:
            print(preview)
            return
        subprocess.run(['docker', 'buildx', 'imagetools', 'create', '-t', target, *references], check=True)
    raw = subprocess.check_output(['docker', 'buildx', 'imagetools', 'inspect', '--raw', target])
    if platforms(json.loads(raw)) != expected:
        raise ValueError('Published manifest does not match the image digests')
    # Resolve the registry's exact index digest rather than hashing formatted JSON.
    descriptor = command('docker', 'buildx', 'imagetools', 'inspect', target)
    digest = next((line.split()[1] for line in descriptor.splitlines() if line.startswith('Digest:')), '')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Missing published index digest')
    output({'image_ref': f'{args.image}@{digest}'})
    print(f'Verified {target} -> {digest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['resolve', 'manifest'])
    parser.add_argument('--config', default=str(Path(__file__).with_suffix('.json')))
    parser.add_argument('--release', required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--source-sha')
    parser.add_argument('--backend', choices=['cpu', 'cuda'])
    parser.add_argument('--image')
    parser.add_argument('--tag')
    parser.add_argument('--architectures')
    parser.add_argument('--digests', default='digests')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.operation == 'manifest':
        for key, env in [('tvm_version', 'TVM_VERSION'), ('ffi_version', 'TVM_FFI_VERSION'),
                         ('tvm_commit', 'TVM_GIT_COMMIT'), ('ffi_commit', 'TVM_FFI_GIT_COMMIT')]:
            config[key] = os.environ.get(env, config[key])
    validate_config(config)
    if args.operation == 'resolve':
        resolve(args, config)
    else:
        manifest(args, config)


def validate_config(config):
    for key in ['tvm_version', 'ffi_version', 'cuda_version']:
        if not re.fullmatch(r'\d+\.\d+\.\d+(?:\.post\d+)?', config[key]):
            raise ValueError('Invalid dependency version: ' + key)
    for key in ['tvm_commit', 'ffi_commit']:
        if not re.fullmatch(r'[0-9a-f]{40}', config[key]):
            raise ValueError('Invalid dependency commit: ' + key)
    if config['cuda_version'] != '13.2.1':
        raise ValueError('Update and validate CUDA build recipes before changing CUDA version')


if __name__ == '__main__':
    main()

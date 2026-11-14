"""Create verified ZIP snapshots from explicit experiment/data release recipes.

No network upload is performed. Running pipelines are gated by their state files.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time
import zipfile

SUFFIXES = {'.pt','.pth','.ckpt','.npz','.json','.jsonl','.yaml','.yml','.csv','.log','.out','.txt','.md','.py','.toml'}
SKIP = {'.git','.venv','__pycache__','wandb','.pytest_cache','.ruff_cache'}
FORBIDDEN = {'.env','token','stored_tokens','.netrc','id_rsa','id_ed25519','authorized_keys'}


def safe_name(name: str) -> str:
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    if any(x in FORBIDDEN or x in SKIP for x in p.parts):
        raise ValueError(f'Excluded artifact path: {name!r}')
    return str(p)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    temp=path.with_name(path.name+'.updating')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def eligible(recipe: dict) -> bool:
    for state in recipe.get('requires_complete',[]):
        p=Path(state)
        if not p.exists() or json.loads(p.read_text()).get('status')!='complete':return False
    return recipe.get('status') not in {'incomplete','excluded'}


def package(recipe: dict, output: Path) -> dict:
    if not eligible(recipe):raise ValueError('Experiment is not complete')
    name=recipe['id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe artifact id')
    safe_name(recipe.get('category','experiments'))
    folder=output/recipe.get('category','experiments')
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(name+'.zip');receipt=folder/(name+'.manifest.json')
    recipe_hash=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    if dest.exists():
        if not receipt.exists():raise FileExistsError(dest)
        old=json.loads(receipt.read_text())
        if old['recipe_sha256']!=recipe_hash or sha256(dest)!=old['zip_sha256']:
            raise ValueError('Existing ZIP differs; use a new release directory/version')
        return old
    partial=dest.with_suffix('.zip.partial')
    if partial.exists():raise FileExistsError(partial)
    records=[];names=set();file_count=0
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        def add(inp, member, source, size=None):
            nonlocal file_count
            member=safe_name(member)
            if member in names:raise ValueError(f'Duplicate ZIP member: {member}')
            names.add(member)
            info=zipfile.ZipInfo(member)
            info.external_attr=(stat.S_IFREG|0o644)<<16
            info.compress_type=zipfile.ZIP_STORED if member.endswith(('.npz','.png','.jpg','.zip')) else zipfile.ZIP_DEFLATED
            info._compresslevel=1
            h=hashlib.sha256();count=0
            with archive.open(info,'w',force_zip64=True) as f:
                for block in iter(lambda:inp.read(4*1024*1024),b''):
                    f.write(block);h.update(block);count+=len(block)
            if size is not None and count!=size:raise ValueError(f'Source size changed: {source}')
            records.append({'path':member,'bytes':count,'sha256':h.hexdigest(),'source':source})
            file_count+=1
            if file_count%10000==0:print(f'{name}: {file_count} files',flush=True)
        def add_file(src,member):
            p=Path(src)
            if p.is_symlink() or not p.is_file():raise ValueError(f'Not a regular source file: {p}')
            before=p.stat()
            with p.open('rb') as f:add(f,member,str(p),before.st_size)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError(f'Source changed during snapshot: {p}')
        for entry in recipe.get('files',[]):add_file(entry['source'],entry['name'])
        for entry in recipe.get('trees',[]):
            base=Path(entry['source'])
            if not base.is_dir():raise FileNotFoundError(base)
            exclude=set(entry.get('exclude_dirs',[]))|SKIP
            for directory,dirs,files in os.walk(base):
                dirs[:]=sorted(d for d in dirs if d not in exclude and not d.startswith('.'))
                for filename in sorted(files):
                    p=Path(directory)/filename
                    if p.name.startswith('.') or p.suffix not in SUFFIXES:continue
                    add_file(p,str(PurePosixPath(entry['prefix'])/p.relative_to(base).as_posix()))
        by_archive={}
        for entry in recipe.get('tar_trees',[]):by_archive.setdefault(entry['source'],[]).append(entry)
        for source,entries in by_archive.items():
            matched=[0]*len(entries)
            with tarfile.open(source,'r|gz') as tar:
                for member in tar:
                    if not member.isfile():continue
                    member_name=member.name.removeprefix('./')
                    for i,entry in enumerate(entries):
                        prefix=entry['member_prefix'].strip('/')
                        if prefix and not member_name.startswith(prefix+'/'):continue
                        rel=member_name[len(prefix)+1:] if prefix else member_name
                        parts=PurePosixPath(rel).parts
                        if any(x.startswith('.') or x in SKIP or x in entry.get('exclude_dirs',[]) for x in parts):continue
                        if PurePosixPath(rel).suffix not in SUFFIXES:continue
                        target=str(PurePosixPath(entry['prefix'])/rel)
                        with tar.extractfile(member) as f:add(f,target,source+'::'+member.name,member.size)
                        matched[i]+=1
            if any(n==0 for n in matched):raise ValueError(f'No selected files in {source}')
        for member,value in recipe.get('json_payloads',{}).items():
            add(io.BytesIO((json.dumps(value,indent=2)+'\n').encode()),member,'generated metadata')
        for member,value in recipe.get('text_payloads',{}).items():
            add(io.BytesIO(value.encode()),member,'release documentation')
        metadata={k:v for k,v in recipe.items() if k not in {'files','trees','tar_trees','json_payloads','text_payloads'}}
        add(io.BytesIO((json.dumps(metadata,indent=2)+'\n').encode()),'ARTIFACT.json','release recipe')
        archive.writestr('FILES.json',json.dumps(records,indent=2))
    with zipfile.ZipFile(partial) as archive:
        bad=archive.testzip()
        if bad:raise ValueError(f'ZIP CRC verification failed: {bad}')
    partial.rename(dest)
    result={'id':name,'category':recipe.get('category','experiments'),'status':recipe.get('status','complete'),
            'zip':str(dest.relative_to(output)),'bytes':dest.stat().st_size,'zip_sha256':sha256(dest),
            'recipe_sha256':recipe_hash,'file_count':len(records),'verified':True,'metadata':metadata}
    write_json(receipt,result)
    print('VERIFIED '+str(dest)+' '+str(result['bytes'])+' bytes',flush=True)
    return result


def verify_extract(archive_path: Path, destination: Path, expected_sha256: str | None = None):
    if expected_sha256 and sha256(archive_path)!=expected_sha256:raise ValueError('ZIP SHA-256 mismatch')
    if destination.exists():raise FileExistsError('Extract to a new directory')
    with zipfile.ZipFile(archive_path) as archive:
        names=set()
        for info in archive.infolist():
            name=safe_name(info.filename)
            if name in names or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Duplicate/symlink member')
            names.add(name)
        files=json.loads(archive.read('FILES.json'))
        if {x['path'] for x in files}|{'FILES.json'}!=names:raise ValueError('Unmanifested ZIP member')
        destination.mkdir(parents=True)
        for entry in files:
            target=destination/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();size=0
            with archive.open(entry['path']) as src,target.open('xb') as dst:
                for block in iter(lambda:src.read(4*1024*1024),b''):
                    dst.write(block);h.update(block);size+=len(block)
            if h.hexdigest()!=entry['sha256'] or size!=entry['bytes']:raise ValueError('Member checksum mismatch')
        (destination/'FILES.json').write_text(json.dumps(files,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('build');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p=sub.add_parser('extract');p.add_argument('archive',type=Path);p.add_argument('--destination',type=Path,required=True);p.add_argument('--sha256')
    args=parser.parse_args()
    if args.action=='extract':verify_extract(args.archive,args.destination,args.sha256);return
    plan=json.loads(args.plan.read_text());args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'builder.lock').open('x') as f:f.write(str(os.getpid()))
    try:
        if hasattr(os,'nice'):os.nice(10)
        cached={}
        while True:
            completed=[];pending=[]
            for recipe in plan['artifacts']:
                if eligible(recipe):
                    if recipe['id'] not in cached:cached[recipe['id']]=package(recipe,args.output)
                    completed.append(cached[recipe['id']])
                else:pending.append(recipe['id'])
            write_json(args.output/'release_index.json',{'release':plan['release'],'artifacts':completed,'pending':pending,'huggingface':plan.get('huggingface',{'status':'repository URL not supplied'})})
            (args.output/'SHA256SUMS').write_text(''.join(x['zip_sha256']+'  '+x['zip']+'\n' for x in completed))
            if not args.watch or not pending:break
            print('WAITING '+', '.join(pending),flush=True);time.sleep(60)
    finally:(args.output/'builder.lock').unlink()


if __name__=='__main__':main()

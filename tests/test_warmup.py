import ast
import asyncio
import base64
import io
import wave
from types import SimpleNamespace
import pytest
from test_generation_error_reporting import load_server_functions, load_app, SERVER


def helpers():
    ns=load_server_functions()
    ns['base64']=base64
    nodes=[node for node in ast.parse(SERVER.read_text()).body
           if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('_warmup_payloads','_run_warmup')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SERVER),'exec'),ns)
    return ns


def test_synthetic_text_and_audio_text_fixtures():
    payloads=helpers()['_warmup_payloads']()
    assert [list(item['type'] for item in p['messages'][0]['content']) for p in payloads]==[['text'],['audio','text']]
    assert all('context' not in p for p in payloads)
    assert len({p['request_id'] for p in payloads})==2
    with wave.open(io.BytesIO(base64.b64decode(payloads[1]['messages'][0]['content'][0]['data'])),'rb') as audio:
        assert (audio.getframerate(),audio.getnchannels(),audio.getnframes())==(16000,1,8000)


@pytest.mark.parametrize('fail',[False,True])
def test_warmup_discards_output_and_reports_errors(fail):
    ns=helpers();requests=[];aborted=[]
    async def stream(sink,sid,model,processor,payload,session_cache):
        requests.append(payload)
        assert session_cache is None
        await sink.emit('token',{'delta':'discarded'})
        await sink.emit('generation_error' if fail else 'generation_complete',{'error':'bad model'})
    async def abort(rid):aborted.append(rid)
    ns['_stream_generate']=stream
    if fail:
        with pytest.raises(RuntimeError,match='bad model'):
            asyncio.run(ns['_run_warmup'](SimpleNamespace(abort=abort),None))
        assert len(aborted)==1
    else:
        assert asyncio.run(ns['_run_warmup'](None,None))=={'ok':True,'passes':['text','audio_text']}
        assert len(requests)==2 and not aborted


def test_warmup_requires_owner_registration():
    async def run():
        socket=load_app(None,None)
        h=socket.handlers
        await h['connect']('guest',{})
        result=await h['warm_up']('guest',{'process_id':'manager','interaction_id':'session'})
        assert result['ok'] is False and 'registered context owner' in result['error']
        await h['disconnect']('guest')
    asyncio.run(run())

from pongoro import pongoro

def test_echo():
    '''
    make sure echo functions correctly
    '''
    
    assert pongoro.echo("hello to pangoro") == 'hello to pangoro', 'incorrect message!'
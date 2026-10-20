"""Synthetic in-domain text for the closed-vocabulary LM. The competition paper (arXiv 2609.03231, App. B)
describes the holdout-style utterances as short conversational sentences built only from the 50 competition
words ("i think so", "we have time", "i am very good", "can i have this", "we can do this"). These sentences
were written by Claude in that style; only the public vocabulary and the paper's examples were used (no
holdout labels). Every sentence is checked to contain only vocabulary words.
  python vocab_corpus.py  -> outputs/vocab_corpus.txt
"""
from pathlib import Path

VOCAB = set("is the a to it i not was we be he that have this they of there and are in but will so all my for "
            "she were any really at out our am it's had him an very has do can time think good always new people "
            "as on".split())

S = """
i think so
we have time
i am very good
i am not good at all
can i have this
we can do this
i think it is good
i think it was a good time
we will be there
we will be there on time
he is not there
she is very good
they are very good people
it is a new time
it's a good time
it's really good
it's not that good
it's all good
it's my time
it's time
it was a very good time
it was not that good
it was really good
i was there
i was not there
i was there for him
i was there all the time
she was there for him
he was there for her
he was there for us
we were there
we were not there
we were all there
they were there
they were not there
they were all there
they were all good people
there is a new one
there is time
there is no time
there is not any time
there was a time
there was a new one
there were people
there were a lot of people
there are people
there are new people
there are always people
there are always new people
there will be time
there will always be time
there will be a new one
i have a new one
i have time
i have no time
i do not have time
i do not have any time
i do not think so
i do think so
i really think so
i really do think so
i always think so
i think so too
she thinks so
he thinks so
we think so
they think so
do you think so
i think that is good
i think that is very good
i think that it is really good
i think this is good
i think this is a good one
i think this is not good
i think we can do this
i think we can do it
i think they can do it
i think he can do it
i think she can do it
i think it will be good
i think it will be a good time
i think it will be new
i think they will be there
i think he will be there
i think she will be there
i think we will be there on time
i think there is time
i think there will be time
i think all of them are good
i think all of us are good
i think all of it is good
i think so but i am not good at it
i am good
i am really good
i am always good
i am not that good
i am very good at this
i am good at it
i am not good at it
i am new
i am new at this
i am new to this
i am in
i am out
i am on it
i am at it
i am so good
i am all in
i am not in
we are good
we are very good
we are all good
we are new
we are new here
we are in
we are out
we are on time
we are not on time
we are all in this
we are in this
we are at it
we are on it
we are so good
they are good
they are very good
they are really good
they are new
they are always there
they are always good
they are not good
they are not that good
they are all new
they are in
they are out
they are on it
they are at it
he is good
he is very good
he is new
he is not new
he is always there
he is really good
he is in
he is out
he is on it
he is at it
he has time
he has a new one
he has no time
he had time
he had a good time
he had a very good time
she is good
she is really good
she is new
she is always there
she is in
she is out
she has time
she has a new one
she had a good time
she had a really good time
she was very good
she was really good
she was always good
he was very good
he was really good
he was always there
he was not there
this is good
this is very good
this is really good
this is not good
this is new
this is my time
this is our time
this is a good time
this is a new time
this is all good
this is it
this is for him
this is for you
this is for us
this is for all of us
that is good
that is very good
that is really good
that is not good
that is new
that is so good
that is it
that is all
that was good
that was very good
that was really good
that was a good time
that was a very good time
that was not good
that was it
that was all
that will be good
that will be all
that will do
can i do this
can i do it
can i have it
can i have that
can i have a new one
can i have time
can i be there
can we do this
can we do it
can we have this
can we have time
can we be there
can he do it
can she do it
can they do it
can they be there
we can do it
we can have this
we can be there
we can be there on time
they can do it
they can do this
they can be there
he can do it
she can do it
i can do it
i can do this
i can do that
i can be there
i can be there on time
i can have it
i will do it
i will do this
i will be there
i will be there on time
i will always be there
i will always be there for him
i will be there for her
i will be there for them
i will be there for you
we will do it
we will do this
we will always be there
we will have time
we will have a good time
they will do it
they will be good
they will have time
he will do it
he will be there
he will be good
she will do it
she will be there
she will be good
it will be good
it will be very good
it will be really good
it will be a good time
it will be new
it will not be good
i have to do it
i have to do this
i have to be there
we have to do it
we have to do this
we have to be there
we have to be there on time
they have to do it
they have to be there
he has to do it
he has to be there
she has to do it
she has to be there
i had to do it
we had to do it
they had to do it
he had to do it
she had to do it
i had a good time
we had a good time
we had a very good time
we had a really good time
they had a good time
i had a new one
we had no time
i had no time
they had no time
do it
do this
do that
do it for him
do it for us
do it all
do it on time
do it at a good time
do not do it
do not do that
do not do this
i do
i do not
we do
we do not
they do
they do not
i do it all the time
i do this all the time
we do it all the time
they do it all the time
i really do
i really do not
i always do
we always do
they always do
he always does
i always have time
we always have time
they always have time
he always has time
she always has time
i always think of him
i always think of her
i always think of you
i think of him
i think of her
i think of them
i think of it
i think of it all the time
i think of this all the time
i think of my people
we think of our people
it is my new one
it is our new one
it is a new one
it is a good one
it is very new
it is really new
it is not new
it is good
it is very good
it is really good
it is not good
it is not that good
it is so good
it is all good
it is all new
it is in
it is out
it is on
it is on time
it is not on time
it is time
it is my time
it is our time
it is a good time
it is not a good time
is it good
is it new
is it on
is it in
is it time
is it there
is it really good
is it really new
is this good
is this new
is this it
is that good
is that new
is that it
is that all
is he there
is she there
are they there
are they good
are they new
are we there
are we on time
are we good
was it good
was it new
was he there
was she there
were they there
were they good
will it be good
will he be there
will she be there
will they be there
will we be there
will it be on time
have we time
has he time
has she time
my time is not new
my time is good
our time is good
our time is new
our people are good
our people are always good
our people are very good
my people are good
my people are always there
all of my people are good
all of our people are good
all of them are good
all of them are new
all of them were there
all of us were there
all of us are here
all of it is good
all of it is new
all of it was good
all of this is good
all of this is new
all people are good
all people are new
people are good
people are new
people are always good
people are really good
people were there
people were good
people will be there
people will always be there
people have time
people do not have time
people think so
people do not think so
people really think so
people always think so
new people are good
new people are always there
new people were there
the people are good
the people are new
the people were there
the people will be there
the people have time
the time is good
the time is new
the time is now
the time was good
the time was not good
the new one is good
the new one is very good
the new one is really good
the new one was good
a new time is here
a good time was had by all
i think it is time
i think it is a good time
i think this is the time
i think that is the time
i think it is not the time
it is not the time
it is the time
this is the time
this is the new one
this is the one
that is the one
that is the new one
that was the one
it was the time
it was the one
i was in
i was out
i was on it
i was at it
i was good
i was very good
i was really good
i was not good
i was new
i was new at this
we were good
we were very good
we were new
we were new at this
we were in
we were out
they were good
they were very good
they were new
they were in
they were out
he was in
he was out
she was in
she was out
i have it
i have this
i have that
i have all of it
i have all of them
i have him
we have him
they have him
she has him
he has it
she has it
they have it
we have it
i have a good one
we have a good one
they have a good one
we have a new one
they have a new one
so it is
so it was
so i think
so we think
so they think
so he is
so she is
so we are
so they are
so it will be
so that is it
so that was it
so this is it
but i think so
but i do not think so
but it is good
but it is not good
but it was good
but it was not good
but they are good
but they are new
but he is good
but she is good
but we are not there
but they were not there
but i was not there
but i am good
but i am not good
and i think so
and it is good
and it was good
and they are good
and we are good
and he is good
and she is good
and we will be there
and they will be there
and that is it
and that is all
and that was it
and that was all
and it is time
and it was time
i think so and it is good
i think so but it is not good
we have time and we can do it
we have time so we can do it
i have time so i can do it
he has time so he can do it
she has time so she can do it
they have time so they can do it
it is good and it is new
it is new and it is good
it is very good and very new
it was good and it was new
we are good and we are new
they are good and they are new
he is good and she is good
she is good and he is good
i am good and we are good
i think it is good and new
i think they are good people
i think he is a good one
i think she is a good one
i think they are all good
i think we are all good
i think we are all in this
i think it is all good
i think it was all good
i think it will all be good
i really think it is good
i really think it will be good
i really think we can do this
i really think they can do it
i always think it is good
i do not think it is good
i do not think it was good
i do not think it will be good
i do not think they are there
i do not think he is there
i do not think she is there
i do not think we can do it
i do not think they can do it
i do not think that is it
i do not have it
i do not have that
i do not have this
i do not have a new one
we do not have it
we do not have time
they do not have time
he does not have time
she does not have time
i am not in
i am not out
i am not on it
i am not new
i am not new at this
i am not that new
we are not new
we are not in
we are not out
they are not in
they are not out
they are not new
he is not in
she is not in
it is not in
it is not out
it is not on
it is not there
it was not there
it will not be there
this is not it
that is not it
that was not it
this was not it
it was a good one
it was a new one
it was not a good one
it was my time
it was our time
it was all good
it was all new
it was so good
it was really new
it was always good
it was always there
it has a new one
it has time
it had time
it had a new one
it will have time
it will always be there
it will always be good
it will always be new
it's new
it's very new
it's really new
it's so good
it's not new
it's not there
it's there
it's on
it's in
it's out
it's on time
it's a new one
it's a good one
it's the one
it's the new one
it's for him
it's for her
it's for us
it's for all of us
it's all new
it's all there
it's always good
it's always there
it's always new
it's really not good
it's not that new
it's my new one
it's our new one
it's our time
it's all of it
it's not all of it
i think it's good
i think it's new
i think it's time
i think it's a good one
i think it's not good
i think it's all good
i think it's really good
i think it's so good
i think it's on time
i think it's there
i think it's in
i think it's out
i think it's for him
i think it's for us
it's good to have time
it's good to be there
it's good to be here
it's good to do this
it's good to have a new one
it's not good to do this
it's time to do it
it's time to do this
it's time to be there
it's a good time to do it
it's a good time to be there
it was time to do it
it was a good time to do it
it was good to be there
it was good to have time
we were there on time
they were there on time
he was there on time
she was there on time
i was there on time
i was always there on time
we are always on time
they are always on time
he is always on time
she is always on time
i am always on time
i am really good at it
he is really good at it
she is really good at it
they are really good at it
we are really good at it
i am not really good at it
he is not good at it
she is not good at it
they are not good at it
we are not good at it
i was good at it
he was good at it
she was good at it
they were good at it
we were good at it
i was not good at it
i will be good at it
we will be good at it
they will be good at it
he will be good at it
she will be good at it
i am very new at this
he is very new at this
she is very new at this
they are very new at this
we are very new at this
i was new to this
we were new to this
they were new to this
he was new to this
she was new to this
any time is good
any time is a good time
any one is good
any of them is good
any of them will do
any of it is good
any of this is good
is there any time
is there any one
is there any new one
was there any time
was there any one
there is not any time
there was not any time
there is not any one
there was not any one
i do not have any
we do not have any
they do not have any
he does not have any
she does not have any
i really do not have any time
do you have any time
do they have any time
does he have any time
does she have any time
have you any time
i will do it at a good time
we will do it at a good time
they will do it at a good time
at that time
at this time
at the time
at any time
at all times
at all
not at all
i am not good at all
it is not good at all
it was not good at all
that is not good at all
that was not good at all
they are not good at all
he is not good at all
she is not good at all
we are not good at all
i do not think so at all
it is not new at all
it was not new at all
as good as new
as good as it is
as good as it was
it is as good as new
it is as good as it was
it was as good as new
it was as good as it is
he is as good as she is
she is as good as he is
they are as good as we are
we are as good as they are
it is not as good as it was
it was not as good as it is
he is not as good as she is
she is not as good as he is
i am as good as him
he is as good as new
as i think
as we think
as they think
as it is
as it was
as it will be
as i was
as we were
as they were
as he was
as she was
as good as that
as new as that
i think of it as new
i think of it as good
i think of him as good
i think of her as good
i think of them as good
i think of us as good
i think of this as good
i think of this as new
i think of that as good
i was there for him and for her
i will be there for him and for her
we will be there for them
they will be there for us
he will be there for us
she will be there for us
i am there for him
i am there for her
i am there for them
i am there for you
we are there for him
we are there for her
we are there for them
they are there for us
they are there for him
he is there for us
she is there for us
he is there for her
she is there for him
he was there for her
she was there for him
they were there for us
we were there for them
i was always there for him
i was always there for her
i will always be there for them
i will always be there for us
we will always be there for them
they will always be there for us
he will always be there for her
she will always be there for him
out of time
out of all of them
out of all of us
out of all of it
i am out of time
we are out of time
they are out of time
he is out of time
she is out of time
it is out of time
we were out of time
they were out of time
i was out of time
he was out of time
she was out of time
we will be out of time
i think we are out of time
i think they are out of time
it is out
it was out
it will be out
the new one is out
the new one will be out
the new one was out
the people are out
the people were out
the people will be out
all of the people were out
all of the people are in
all of the people will be in
i am in for it
we are in for it
they are in for it
he is in for it
she is in for it
i was in for it
we were in for it
it is in there
it was in there
it will be in there
it is on there
it was on there
it is on the new one
it was on the new one
it is in the new one
it was in the new one
it is all in there
it is all on there
it was all in there
it was all on there
i have it on
we have it on
they have it on
he has it on
she has it on
i had it on
he had it on
she had it on
i think it is on
i think it is in
i think it is out
i think it was on
i think it was in
i think it was out
we are on
they are on
he is on
she is on
i am on
i was on
we were on
they were on
he was on
she was on
we are on it all the time
they are on it all the time
i am on it all the time
he is on it all the time
she is on it all the time
"""

def sentences():
    out, bad = [], []
    for line in S.strip().splitlines():
        s = line.strip().lower()
        if not s:
            continue
        (out if all(w in VOCAB for w in s.split()) else bad).append(s)
    return out, bad


if __name__ == "__main__":
    out, bad = sentences()
    p = Path(__file__).resolve().parents[1] / "outputs" / "vocab_corpus.txt"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(len(out), "sentences kept;", len(bad), "dropped (non-vocabulary words):", bad[:20])

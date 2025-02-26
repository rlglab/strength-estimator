import sys
import board as bd
from tqdm import tqdm
import re
import os

game = []
game_flag = False
valid_players = {}
min_blitz, max_blitz = 0, 3500
total_blitz_games = 0
blitz_rating = [0 for i in range((max_blitz - min_blitz)//100 + 1)]
game_cnt = 0
t_cnt = 0
total_t_cnt = 0

def eloNotIn(min_elo, max_elo, elo):
    return elo <= min_elo or elo > max_elo

def whiteID(move): 
    if 'O-O-O' in move:
        return bd.actLongCastle('w')
    if 'O-O' in move:
        return bd.actShortCastle('w')
    if move[0].islower():
        return bd.actPawn('P', move)
    if move[0] == 'N':
        bd.findPinnedPieces('W')
        return bd.actKnight('N', move)
    if move[0] == 'K':
        return bd.actKing('K', move)
    bd.findPinnedPieces('W')
    return bd.actBRQ(move[0], move)

def blackID(move):
    if 'O-O-O' in move:
        return bd.actLongCastle('b')
    if 'O-O' in move:
        return bd.actShortCastle('b')
    if move[0].islower():
        return bd.actPawn('p', move)
    if move[0] == 'N':
        bd.findPinnedPieces('B')
        return bd.actKnight('n', move)
    if move[0] == 'K':
        return bd.actKing('k', move)
    bd.findPinnedPieces('B')
    return bd.actBRQ(move[0].lower(), move)

def appendAGame(append_line):
    global game_flag
    line = append_line.replace('\n', '')
    if 'Event' in line[:6]:
        game.append(line)
    elif 'UTCDate' in line[:8]:
        game.append(line)
    elif 'TimeControl' in line[:12]:
        game.append(line)
    elif 'Date' not in line and '1.' in line:
        game.append(line)
        if '1-0' in line:
            game.append(1.0)
        elif '0-1' in line:
            game.append(-1.0)
        else:
            game.append(0.0)
        game_flag = True
    elif 'WhiteRatingDiff' not in line[:16] and 'BlackRatingDiff' not in line[:16] and\
         'WhiteTitle' not in line[:11] and 'BlackTitle' not in line[:11]:
        if 'White' in line[:6] or 'Black' in line[:6]:
            game.append(line)


def convertGame(game):
    global total_blitz_games
    global t_cnt, total_t_cnt
    training_format = ''
    result = game[-1]
    event = 'Blitz'
    temp_wr, temp_br = -1, -1
    temp_pw, temp_pb = '', ''
    base_time = 0
    add_sec = 0
    for game_info in game[:-1]:
        # print('game_info:', game_info)
        if 'Event' in game_info[:6]:
            if 'Blitz' in game_info:
                training_format = f';GM[chess]RE[{result}]EV[Blitz]'
            else:
                return 
        elif 'UTCDate' in game_info[:8]:
            date = game_info.split('"')[1]
            training_format += f'DT[{date}]'
        elif 'WhiteElo' in game_info[:9] or 'BlackElo' in game_info[:9]:
            rating = game_info.split('"')[1]
            if rating == '?' or rating == '':
                return
            else:
                if event == 'Blitz':
                    if eloNotIn(min_blitz, max_blitz, int(rating)):
                        return
                else:
                    return
            if 'WhiteElo' in game_info:
                temp_wr = int(rating)
                training_format += f'WR[{rating}]'
            else:
                temp_br = int(rating)
                training_format += f'BR[{rating}]'
        elif 'White' in game_info[:6] or 'Black' in game_info[:6]:
            player = game_info.split('"')[1]
            if player == '?' or player == '':
                return
            if 'White' in game_info[:6]:
                assert('PW[' not in training_format)
                temp_pw = player
                training_format += f'PW[{player}]'
            else:
                assert('PB[' not in training_format)
                temp_pb = player
                training_format += f'PB[{player}]'
        elif 'TimeControl' in game_info[:12]:
            time_control = game_info.split('"')[1]
            base_time = int(time_control.split('+')[0])
            add_sec = int(time_control.split('+')[1])
            # print('time setting:', base_time, add_sec)
            # exit()
        elif '1. ' in game_info:
            move_cnt = 0
            bd.resetBoard()
            moves = game_info.split(' ')
            turn = 'B'
            time_left = {}
            time_left['B'] = base_time - add_sec
            time_left['W'] = base_time - add_sec
            for i in range(len(moves[:-1])):
                action_id = -1
                move = moves[i]
                # print(move)
                pattern = r'(\d+):(\d+):(\d+)\]'
                match = re.match(pattern, move)
                if match:
                    # exit()
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = int(match.group(3))
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    
                    training_format += f'TM[{total_seconds}]'
                    # time_left[turn] += add_sec
                    # training_format += f'TM[{time_left[turn] - total_seconds}]'

                    # total_t_cnt += 1
                    # if time_left[turn] - total_seconds >= 30:
                    #     t_cnt += 1
                    # if total_t_cnt > 10000000:
                    #     print('total move:', total_t_cnt)
                    #     print('accept move: ',t_cnt)
                    #     print('accept_ratio:', t_cnt / total_t_cnt)
                    #     exit()
                    

                    # time_left[turn] = total_seconds
                    continue
                newmove = move.replace('?', '')
                newmove = newmove.replace('!', '')
                if not move[0].isalpha():
                    continue
                if turn == 'B':
                    action_id = whiteID(newmove)
                    training_format += f';B[{action_id}]'
                    turn = 'W'
                else:
                    action_id = blackID(newmove)
                    training_format += f';W[{action_id}]'
                    turn = 'B'
                    move_cnt += 1
            if move_cnt <= 10:
                return
    print(f'({training_format})')
    if event == 'Blitz':
        total_blitz_games += 1
        blitz_rating[(temp_wr - min_blitz)//100] += 1
        blitz_rating[(temp_br - min_blitz)//100] += 1
    if valid_players.get(temp_pw) != None:
        valid_players[temp_pw] = valid_players[temp_pw] + 1
    else:
        valid_players[temp_pw] = 1
    if valid_players.get(temp_pb) != None:
        valid_players[temp_pb] = valid_players[temp_pb] + 1
    else:
        valid_players[temp_pb] = 1

YEAR = sys.argv[1]
MONTH = sys.argv[2]
print(f'year: {YEAR}, month: {MONTH}', file=sys.stderr)
date = False

if not os.path.exists(f'database{YEAR}/{YEAR}{MONTH}'):
    os.makedirs(f'database{YEAR}/{YEAR}{MONTH}')
    print(f'create database{YEAR}/{YEAR}{MONTH}')

with open(f'database{YEAR}/lichess_db_standard_rated_{YEAR}-{MONTH}.pgn', 'r') as pgn:
    for line in tqdm(pgn):
        appendAGame(line)
        if game_flag:
            convertGame(game)
            game_cnt += 1
            if total_blitz_games % 10000000 == 9999999:
                print(f'total_blitz_games: {total_blitz_games}', file=sys.stderr)  
            game_flag = False
            game.clear() 

with open(f'database{YEAR}/{YEAR}{MONTH}/{YEAR}-{MONTH}-players.txt', 'w') as txt:
    games_over_100, games_over_500, games_over_1000 = 0, 0, 0
    for player, num in valid_players.items():
        if num >= 1000:
            games_over_100 += 1
            games_over_500 += 1
            games_over_1000 += 1
        elif num >= 500:
            games_over_100 += 1
            games_over_500 += 1
        elif num >= 100:
            games_over_100 += 1
        txt.write(f'{player} {num}\n') 
    txt.write(f'100+: {games_over_100}\n') 
    txt.write(f'500+: {games_over_500}\n') 
    txt.write(f'1000+: {games_over_1000}\n') 

with open(f'database{YEAR}/{YEAR}{MONTH}/{YEAR}-{MONTH}-result.txt', 'w') as txt:
    txt.write(f'# players: {len(valid_players)}\n')
    txt.write(f'# blitz games: {total_blitz_games}\n')
    now = min_blitz
    s = ''
    for game in blitz_rating:
        s += f'{now}-{now + 99}:\t{game}\t{game / total_blitz_games * 100:.2f}\n'
        now += 100
    txt.write(f'rating distribution: {min_blitz}-{max_blitz}\n')
    txt.write(f'{s}\n')

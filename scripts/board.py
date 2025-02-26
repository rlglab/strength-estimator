assert_flag = False
square = {'a1':0, 'b1':1, 'c1':2, 'd1':3, 'e1':4, 'f1':5, 'g1':6, 'h1':7,
          'a2':8, 'b2':9, 'c2':10, 'd2':11, 'e2':12, 'f2':13, 'g2':14, 'h2':15,
          'a3':16, 'b3':17, 'c3':18, 'd3':19, 'e3':20, 'f3':21, 'g3':22, 'h3':23,
          'a4':24, 'b4':25, 'c4':26, 'd4':27, 'e4':28, 'f4':29, 'g4':30, 'h4':31,
          'a5':32, 'b5':33, 'c5':34, 'd5':35, 'e5':36, 'f5':37, 'g5':38, 'h5':39,
          'a6':40, 'b6':41, 'c6':42, 'd6':43, 'e6':44, 'f6':45, 'g6':46, 'h6':47,
          'a7':48, 'b7':49, 'c7':50, 'd7':51, 'e7':52, 'f7':53, 'g7':54, 'h7':55,
          'a8':56, 'b8':57, 'c8':58, 'd8':59, 'e8':60, 'f8':61, 'g8':62, 'h8':63}

squareIDtoSTR = {value: key for key, value in square.items()}

piece = {'K':'\u265A', 'Q':'\u265B', 'R':'\u265C', 'B':'\u265D', 'N':'\u265E', 'P':'\u265F',
         'k':'\u2654', 'q':'\u2655', 'r':'\u2656', 'b':'\u2657', 'n':'\u2658', 'p':'\u2659',
         '-':'-'}

board = [i for i in 'RNBQKBNRPPPPPPPP--------------------------------pppppppprnbqkbnr']
king_pos = [4, 60]
pinned = [False for i in range(64)]

# (Δx, Δy)
direction = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
knight_direction = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# action_id = square_id(0-63) * 73 + move_id(0-72), (Δx, Δy)
move_id = {(0, 1):0,  (1, 1):1,  (1, 0):2,   (1, -1):3,  (0, -1):4,   (-1, -1):5,  (-1, 0):6,  (-1, 1):7,
           (0, 2):8,  (2, 2):9,  (2, 0):10,  (2, -2):11, (0, -2):12,  (-2, -2):13, (-2, 0):14, (-2, 2):15,
           (0, 3):16, (3, 3):17, (3, 0):18,  (3, -3):19, (0, -3):20,  (-3, -3):21, (-3, 0):22, (-3, 3):23,
           (0, 4):24, (4, 4):25, (4, 0):26,  (4, -4):27, (0, -4):28,  (-4, -4):29, (-4, 0):30, (-4, 4):31,
           (0, 5):32, (5, 5):33, (5, 0):34,  (5, -5):35, (0, -5):36,  (-5, -5):37, (-5, 0):38, (-5, 5):39,
           (0, 6):40, (6, 6):41, (6, 0):42,  (6, -6):43, (0, -6):44,  (-6, -6):45, (-6, 0):46, (-6, 6):47,
           (0, 7):48, (7, 7):49, (7, 0):50,  (7, -7):51, (0, -7):52,  (-7, -7):53, (-7, 0):54, (-7, 7):55,
           (1, 2):56, (2, 1):57, (2, -1):58, (1, -2):59, (-1, -2):60, (-2, -1):61, (-2, 1):62, (-1, 2):63}

promote_id = {(-1, 'R'):64, (-1, 'B'):65, (-1, 'N'):66, 
               (0, 'R'):67,  (0, 'B'):68,  (0, 'N'):69, 
               (1, 'R'):70,  (1, 'B'):71,  (1, 'N'):72}

def square2RC(square): #str -> 0-7
    return (int(square[1]) - 1, ord(square[0]) - ord('a')) 

def squareID2RC(square):
    return square // 8, square % 8

def removeCheckTakeMate(move, check=True, take=True):
    if check:
        newmove = move.replace('+', '')
        newmove = newmove.replace('#', '')
    if take:
        newmove = newmove.replace('x', '')
    return newmove

def outOfBoard(row, col):
    return row >= 8 or row < 0 or col >= 8 or col < 0

def colrow2Square(col, row): #0-7
    return square[chr(ord('a') + col) + str(row + 1)]

def showBoard():
    for i in range(7, -1, -1):
        output = str(i + 1)
        for b in board[i*8:(i+1)*8]:
            # output = output + '  ' + piece[b]
            output = output + '  ' + b
        print(output)
    print('   a  b  c  d  e  f  g  h')
    print()

def resetBoard(): # uppercase: white, lowercase: black
    global board, pinned, king_pos
    board = [i for i in 'RNBQKBNRPPPPPPPP--------------------------------pppppppprnbqkbnr']
    pinned = [False for i in range(64)]
    king_pos = [4, 60]

def findPinnedPieces(turn, debug=False):
    global pinned
    pinned = [False for i in range(64)]
    if turn == 'W':
        king_row, king_col = squareID2RC(king_pos[0])
        opp_b, opp_r, opp_q = 'b', 'r', 'q'
    else:
        king_row, king_col = squareID2RC(king_pos[1])
        opp_b, opp_r, opp_q = 'B', 'R', 'Q'
    for b_dir in [direction[i] for i in range(1, 8, 2)]:
        potential_pin = -1
        attack_piece_found = False
        # if debug:
        #     print('dir (col, row):', b_dir)
        for j in range(1, 8):
            piece_row, piece_col = king_row + b_dir[1]*j, king_col + b_dir[0]*j
            if outOfBoard(piece_row, piece_col):
                break
            check_piece = board[colrow2Square(piece_col, piece_row)]
            # if debug:
            #     print(f'check: {check_piece}, potential pin: {potential_pin}, attack found: {attack_piece_found}')
            if turn == 'W' and check_piece.isupper() or turn == 'B' and check_piece.islower():
                if potential_pin != -1:
                    break
                elif not attack_piece_found:
                    potential_pin = colrow2Square(piece_col, piece_row)
            elif check_piece == opp_b or check_piece == opp_q:
                if potential_pin != -1:
                    pinned[potential_pin] = True
                break
            elif check_piece != '-':
                break
            
    for r_dir in [direction[i] for i in range(0, 8, 2)]:
        potential_pin = -1
        attack_piece_found = False
        # if debug:
        #     print('dir (col, row):', r_dir)
        for j in range(1, 8):
            piece_row, piece_col = king_row + r_dir[1]*j, king_col + r_dir[0]*j
            if outOfBoard(piece_row, piece_col):
                break
            check_piece = board[colrow2Square(piece_col, piece_row)]
            # if debug:
            #     print(f'check: {check_piece}, potential pin: {potential_pin}, attack found: {attack_piece_found}')
            if turn == 'W' and check_piece.isupper() or turn == 'B' and check_piece.islower():
                if potential_pin != -1:
                    break
                elif not attack_piece_found:
                    potential_pin = colrow2Square(piece_col, piece_row)
            elif check_piece == opp_r or check_piece == opp_q:
                if potential_pin != -1:
                    pinned[potential_pin] = True
                break
            elif check_piece != '-':
                break
    
def actLongCastle(turn): # 0-0-0
    if turn == 'w':
        board[:5] = list('--KR-')
        # print(f'0-0-0[{4 * 73 + 12}]')
        # showBoard()
        king_pos[0] = 2
        return 'e1a1'
        return 4 * 73 + 14
    else:
        board[56:61] = list('--kr-')
        # print(f'0-0-0[{60 * 73 + 12}]')
        # showBoard()
        king_pos[1] = 58
        return 'e8a8'
        return 60 * 73 + 14

def actShortCastle(turn): # 0-0
    if turn == 'w':
        board[4:8] = list('-RK-')
        # print(f'0-0[{4 * 73 + 10}]')
        # showBoard()
        king_pos[0] = 6
        return 'e1h1'
        return 4 * 73 + 10
    else:
        board[60:] = list('-rk-')
        # print(f'0-0[{60 * 73 + 10}]')
        # showBoard()
        king_pos[1] = 62
        return 'e8h8'
        return 60 * 73 + 10

def actPawn(pawn, move):
    newmove = removeCheckTakeMate(move, check=True, take=False)
    from_col = newmove[0]
    if pawn == 'P':
        del_row = 1
        enpassant_row = 4
        moveto_row = '4'
        forward_id = 0
    else:
        del_row = -1
        enpassant_row = 3
        moveto_row = '5'
        forward_id = 4
    if '=' not in newmove:
        move_to = square[newmove[-2:]]
        move_from = square[from_col + str(int(newmove[-1])-del_row)]
        if 'x' not in newmove:
            action_id = forward_id
            if newmove[-1] == moveto_row and board[square[from_col + str(int(newmove[-1])-del_row)]] == '-':
                move_from = square[from_col + str(int(newmove[-1])-2*del_row)]
                action_id = forward_id + 8
        else:
            # print((ord(newmove[-2]) - ord(from_col), del_row))
            action_id = move_id[(ord(newmove[-2]) - ord(from_col), del_row)]
    else:
        move_to = square[newmove[-4:-2]]
        move_from = square[from_col + str(int(newmove[-3])-del_row)]
    # print(f'{move}, from: {move_from}, to: {move_to}')
    board[move_from] = '-'
    if move_from // 8 == enpassant_row and 'x' in move and board[move_to] == '-':
        # print('enpassant')
        board[colrow2Square(move_to % 8, move_from // 8)] = '-'
    board[move_to] = pawn
    ret_str = squareIDtoSTR[move_from] + squareIDtoSTR[move_to]
    if '=' in newmove:
        if pawn == 'P':
            board[move_to] = newmove[-1]
        else:
            board[move_to] = newmove[-1].lower()
        if 'N' not in newmove:
            ret_str += newmove[-1].lower()
        # -1/rbn: 64-66, 0/rbn: 67-69, 1/rbn: 70-72
        if 'Q' not in newmove:
            action_id = promote_id[(move_to % 8 - move_from % 8, newmove[-1])]
        else:
            action_id = move_id[(move_to % 8 - move_from % 8, move_to // 8 - move_from // 8)]
    # print(f'{move}[{move_from * 73 + action_id}]')
    # showBoard()
    return ret_str
    return move_from * 73 + action_id

def actKnight(knight, move):
    newmove = removeCheckTakeMate(move)
    move_to = square[newmove[-2:]]
    move_from = -1
    to_row, to_col = square2RC(newmove[-2:])
    if len(newmove) == 3:
        for dir in knight_direction:
            from_col, from_row = to_col + dir[0], to_row + dir[1]
            if outOfBoard(from_col, from_row):
                continue
            else:
                if board[colrow2Square(from_col, from_row)] == knight:
                    if pinned[colrow2Square(from_col, from_row)]:
                        continue
                    move_from = colrow2Square(from_col, from_row)
                    action_id = move_id[(to_col - from_col, to_row - from_row)]
                    break
    elif len(newmove) == 4:
        from_line = newmove[1]
        for dir in knight_direction:
            from_col, from_row = to_col + dir[0], to_row + dir[1]
            if outOfBoard(from_col, from_row):
                continue
            else:
                if from_line.isdigit():
                    if from_row != ord(from_line) - ord('1'):
                        continue
                elif from_line.isalpha():
                    if from_col != ord(from_line) - ord('a'):
                        continue
                if board[colrow2Square(from_col, from_row)] == knight:
                    if pinned[colrow2Square(from_col, from_row)]:
                        continue
                    move_from = colrow2Square(from_col, from_row)
                    action_id = move_id[(to_col - from_col, to_row - from_row)]
                    break
    elif  len(newmove) == 5:
        from_square = square[newmove[1:3]]  
        for dir in knight_direction:
            from_col, from_row = to_col + dir[0], to_row + dir[1]
            if outOfBoard(from_col, from_row):
                continue
            if colrow2Square(from_col, from_row) != from_square:
                continue
            else:
                if board[from_square] == knight:
                    move_from = from_square
                    action_id = move_id[(to_col - from_col, to_row - from_row)]

    if move_from == -1:
        showBoard()
        print(f'actKnight: move_from = -1, move[{move}], newmove[{newmove}]')
        assert move_from != -1, f'actKnight: move_from = -1, move[{move}], newmove[{newmove}]'
    board[move_from] = '-'
    board[move_to] = knight
    ret_str = squareIDtoSTR[move_from] + squareIDtoSTR[move_to]
    return ret_str
    # print(f'{move}[{move_from * 73 + action_id}]')
    # showBoard()
    return move_from * 73 + action_id

def actBRQ(brq, move):
    newmove = removeCheckTakeMate(move)
    move_to = square[newmove[-2:]]
    move_from = -1
    to_row, to_col = square2RC(newmove[-2:])
    if brq in 'bB':
        target_direction = [direction[i] for i in range(1, 8, 2)]
    elif brq in 'rR':
        target_direction = [direction[i] for i in range(0, 8, 2)]
    else: 
        target_direction = [direction[i] for i in range(8)]
    for dir in target_direction:
        for i in range(1, 8):
            from_col, from_row = to_col + dir[0]*i, to_row + dir[1]*i
            if outOfBoard(from_col, from_row) or (board[colrow2Square(from_col, from_row)] != '-' and board[colrow2Square(from_col, from_row)] != brq):
                break
            if len(newmove) == 4:
                from_line = newmove[1]
                if from_line.isdigit():
                    if from_row != ord(from_line) - ord('1'):
                        if board[colrow2Square(from_col, from_row)] == brq:
                            break
                        continue
                elif from_line.isalpha():
                    if from_col != ord(from_line) - ord('a'):
                        if board[colrow2Square(from_col, from_row)] == brq:
                            break
                        continue
            elif len(newmove) == 5:
                from_square = square[newmove[1:3]]  
                if colrow2Square(from_col, from_row) != from_square:
                    continue
            if board[colrow2Square(from_col, from_row)] == brq:
                if pinned[colrow2Square(from_col, from_row)] == True:
                    if brq.isupper():
                        king_row, king_col = squareID2RC(king_pos[0])
                    else:
                        king_row, king_col = squareID2RC(king_pos[1])

                    dir_kf = (king_col - from_col, king_row - from_row)
                    if dir_kf[0] == 0:
                        if not dir[0] == 0:
                            break
                    elif dir_kf[1] == 0:
                        if not dir[1] == 0:
                            break
                    elif dir_kf[0] * dir_kf[1] > 0:
                        if not dir[0] * dir[1] > 0:
                            break
                    else:
                        if not dir[0] * dir[1] < 0:
                            break
        
                move_from = colrow2Square(from_col, from_row)
                action_id = move_id[(to_col - from_col, to_row - from_row)]
                break
        if move_from != -1:
            break
    if move_from == -1:
        showBoard()
        print(f'actBRQ: move_from = -1, move[{move}], newmove[{newmove}]')
        assert move_from != -1, f'actBRQ: move_from = -1, move[{move}], newmove[{newmove}]'
    board[move_from] = '-'
    board[move_to] = brq
    
    ret_str = squareIDtoSTR[move_from] + squareIDtoSTR[move_to]
    return ret_str
    # print(f'{move}[{move_from * 73 + action_id}]')
    # showBoard()
    return move_from * 73 + action_id
    
def actKing(king, move):
    newmove = removeCheckTakeMate(move)
    move_to = square[newmove[1:3]]
    move_from = -1
    to_row, to_col = square2RC(newmove[1:3])
    for dir in direction:
        from_col, from_row = to_col + dir[0], to_row + dir[1]
        if outOfBoard(from_col, from_row):
            continue
        else:
            if board[colrow2Square(from_col, from_row)] == king:
                move_from = colrow2Square(from_col, from_row)
                action_id = move_id[(to_col - from_col, to_row - from_row)]
                break
    if move_from == -1:
        showBoard()
        assert move_from != -1, f'actKing: move_from = -1, move[{move}]'
    board[move_from] = '-'
    board[move_to] = king
    if king == 'K':
        king_pos[0] = move_to
    else:
        king_pos[1] = move_to
    # print(f'{move}[{move_from * 73 + action_id}]')
    # showBoard()
        
    ret_str = squareIDtoSTR[move_from] + squareIDtoSTR[move_to]
    return ret_str
    return move_from * 73 + action_id


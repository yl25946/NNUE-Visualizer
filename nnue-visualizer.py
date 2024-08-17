from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import chess


# different NNUE quantizations
SCALE = 400
L1Q = 255
OutputQ = 64

# NNUE Evaluation


class NNUE:
    def __init__(self, filename, feature_size: int = 768, hidden_size: int = 2048, output_buckets: int = 8):
        data = Path(filename).read_bytes()
        self.raw = np.frombuffer(data, dtype='<i2')
        self.ft = self.raw[:feature_size *
                           hidden_size].reshape(feature_size, hidden_size)
        self.ftBiases = self.raw[feature_size * hidden_size: feature_size *
                                 hidden_size + hidden_size].reshape(hidden_size)
        self.outputWeights = self.raw[feature_size * hidden_size +
                                      hidden_size: feature_size * hidden_size +
                                      hidden_size + hidden_size * 2 * output_buckets].reshape(2, hidden_size, output_buckets)
        self.outputBias = self.raw[feature_size *
                                   hidden_size + hidden_size + hidden_size * 2 * output_buckets: feature_size *
                                   hidden_size + hidden_size + hidden_size * 2 * output_buckets + output_buckets].reshape(output_buckets)

        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_buckets = output_buckets

    def feature_index(self, piece: chess.Piece, square: chess.Square, flipped: bool):
        if flipped:
            side_is_black = piece.color
            square = square ^ 0x38
        else:
            side_is_black = not piece.color
        return square + int(piece.piece_type - 1) * 64 + (384 if side_is_black else 0)

    # Uses linear buckets
    def get_output_bucket(self, board: chess.Board):
        piece_count = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_count += 1

        divisor = int((32 + self.output_buckets - 1) / self.output_buckets)

        return int((piece_count - 2) / divisor)

    # visualizes the weights of a specific neuron for a specific board
    def visualize1(self, board: chess.Board, ax, font_size: int = 6, neuron_index: int = 0):

        intensity: np.ndarray[np.int32] = np.zeros((8, 8), dtype=np.int32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            intensity[7-chess.square_rank(square)][chess.square_file(
                square)] = self.ft[self.feature_index(piece, square, False)][neuron_index]

        self.display(intensity, ax, str(neuron_index), font_size)

    # visualizes the weights of a specific neuron for a specific piece type and color
    def visualize2(self, piecetype, color, ax, font_size: int = 6, neuron_index: int = 0):

        intensity: np.ndarray[np.int32] = np.zeros((8, 8), dtype=np.int32)

        for square in chess.SQUARES:
            piece = chess.Piece(piecetype, color)
            if piece is None:
                continue
            intensity[7-chess.square_rank(square)][chess.square_file(
                square)] = self.ft[self.feature_index(piece, square, False)][neuron_index]

        self.display(intensity, ax, str(neuron_index), font_size),

    # visualizes the kings of a board
    def visualize3(self, board: chess.Board, ax, font_size: int = 6,):

        intensity = np.zeros((8, 8))

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            if piece.piece_type == chess.KING:
                intensity[7-chess.square_rank(square)
                          ][chess.square_file(square)] = 1

        self.display(intensity, ax, font_size)

    def display(self, intensity: np.ndarray[np.int32], ax, x_label: str | None = None, font_size: int = 6):
        ax.imshow(intensity, cmap="magma", interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(x_label, fontsize=font_size)
        fig.tight_layout()

    def full_evaluate(self, board: chess.Board):
        output_bucket = self.get_output_bucket(board)
        accumulatorWhite = self.ftBiases.copy()
        accumulatorBlack = self.ftBiases.copy()

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            accumulatorWhite += self.ft[self.feature_index(
                piece, square, False)]
            accumulatorBlack += self.ft[self.feature_index(
                piece, square, True)]

        if board.turn == chess.WHITE:
            total = (np.sum(accumulatorWhite.clip(0, L1Q).astype(np.int32) ** 2 * self.outputWeights[chess.WHITE][self.hidden_size][output_bucket])
                     + np.sum(accumulatorBlack.clip(0, L1Q).astype(np.int32) ** 2 * self.outputWeights[chess.BLACK][self.hidden_size][output_bucket]))
        else:
            total = (np.sum(accumulatorBlack.clip(0, L1Q).astype(np.int32) ** 2 * self.outputWeights[chess.WHITE][self.hidden_size][output_bucket])
                     + np.sum(accumulatorWhite.clip(0, L1Q).astype(np.int32) ** 2 * self.outputWeights[chess.BLACK][self.hidden_size][output_bucket]))

        value = (
            total // L1Q + self.outputBias[output_bucket]) * SCALE // (L1Q * OutputQ)

        return value


network = NNUE("2048-8.bin")
# network.visualize1(chess.Board(), 1)
# network.visualize2(chess.QUEEN, chess.WHITE, 604)
# network.visualize2(chess.ROOK, chess.WHITE, 109)
# network.visualize2(chess.ROOK, chess.WHITE, 153)
# for i in range(0, 2048):
#     # network.visualize1(chess.Board(chess.STARTING_BOARD_FEN), i)
#     # network.visualize2(chess.PAWN, chess.WHITE, i)
#     network.visualize3(chess.Board('8/4k3/8/8/5K2/8/8/8 w - - 0 1'))
#     pass

# fig, ax = plt. subplots(1, 1)
# network.visualize2(chess.PAWN, chess.WHITE, ax, 16, 272)
# fig.savefig(
#     "tmp.png",
#     pad_inches=0,
#     transparent=False,
#     dpi=1200,
# )


# how many neurons wide we want the visualization to be
x = 8
# how many neurons tall we want the visualization to be
y = 8
# how big you want the font to be
font_size = 6

fig, ax = plt.subplots(x, y)

for i in range(x):
    for j in range(y):
        neuron = i * x + j
        network.visualize2(chess.PAWN, chess.WHITE,
                           ax[i, j], font_size, neuron)

fig.savefig('tmp.png',
            format='png',
            pad_inches=0,
            transparent=False,
            dpi=1200,
            )


plt.close()
# network.full_evaluate(chess.Board())
# network.full_evaluate(chess.Board(
#     "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 1"))
